#!/usr/bin/env python3
"""Find and test cameras before wiring them into a site config.

Three modes (stdlib + cv2 only — no extra deps):

  # 1. Test one RTSP/HTTP URL works (connects, resolution, fps, saves a snapshot)
  python tools/camera_probe.py --test "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/102" --snapshot cam1.jpg

  # 2. Scan a subnet for devices with the RTSP port open (find camera IPs)
  python tools/camera_probe.py --scan 192.168.1.0/24

  # 3. ONVIF WS-Discovery — ask cameras on the LAN to announce themselves
  python tools/camera_probe.py --discover
"""
from __future__ import annotations

import argparse
import socket


def test_url(url: str, snapshot: str | None) -> int:
    import cv2
    print(f"[probe] opening {url.split('@')[-1]} …")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[probe] FAILED to open. Check: IP reachable? creds right? path correct? on same network?")
        return 1
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[probe] opened but no frame decoded — wrong stream path or codec issue.")
        cap.release()
        return 1
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"[probe] OK — {w}x{h} @ {fps:.0f} fps")
    if snapshot:
        cv2.imwrite(snapshot, frame)
        print(f"[probe] saved a frame to {snapshot} — open it to confirm it's the right camera.")
    cap.release()
    return 0


def scan_subnet(cidr: str, port: int, timeout: float) -> int:
    import ipaddress
    from concurrent.futures import ThreadPoolExecutor
    hosts = [str(h) for h in ipaddress.ip_network(cidr, strict=False).hosts()]
    print(f"[probe] scanning {len(hosts)} hosts on {cidr} for port {port} …")

    def check(ip: str):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            return ip if s.connect_ex((ip, port)) == 0 else None
        finally:
            s.close()

    found = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        for ip in ex.map(check, hosts):
            if ip:
                found.append(ip)
                print(f"  [+] {ip}:{port} open  →  likely a camera. Test: "
                      f'--test "rtsp://user:pass@{ip}:{port}/<stream-path>"')
    if not found:
        print("[probe] no open RTSP ports found. Cameras on a different subnet/VLAN, or using another port?")
    return 0


def discover(timeout: float) -> int:
    import re
    import uuid
    msg = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
        'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
        'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f'<e:Header><w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>'
        '<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
        '<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>'
        '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>'
    )
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    s.settimeout(timeout)
    print("[probe] sending ONVIF WS-Discovery probe (multicast 239.255.255.250:3702) …")
    s.sendto(msg.encode(), ("239.255.255.250", 3702))
    found = set()
    try:
        while True:
            data, addr = s.recvfrom(65535)
            for x in re.findall(r"https?://[^\s<>]+", data.decode(errors="ignore")):
                found.add((addr[0], x))
    except socket.timeout:
        pass
    finally:
        s.close()
    if not found:
        print("[probe] no ONVIF cameras answered. Try --scan, or the camera's app/router page for its IP.")
    for ip, xaddr in sorted(found):
        print(f"  [+] ONVIF device at {ip}  service: {xaddr}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Find + test cameras before wiring them into a site config.")
    p.add_argument("--test", metavar="URL", help="Open an RTSP/HTTP URL and report + snapshot.")
    p.add_argument("--snapshot", default="", help="Save the test frame here (with --test).")
    p.add_argument("--scan", metavar="CIDR", help="Scan a subnet for open RTSP ports, e.g. 192.168.1.0/24")
    p.add_argument("--port", type=int, default=554)
    p.add_argument("--discover", action="store_true", help="ONVIF WS-Discovery on the LAN.")
    p.add_argument("--timeout", type=float, default=1.0, help="Per-host / discovery timeout (s).")
    args = p.parse_args()

    if args.test:
        raise SystemExit(test_url(args.test, args.snapshot or None))
    if args.scan:
        raise SystemExit(scan_subnet(args.scan, args.port, args.timeout))
    if args.discover:
        raise SystemExit(discover(max(args.timeout, 3.0)))
    p.error("pick one: --test URL | --scan CIDR | --discover")


if __name__ == "__main__":
    main()
