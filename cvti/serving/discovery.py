"""Find the cameras instead of asking the installer to know them (EP-05-T4).

Two pieces, both stdlib-only so they run inside the bundle:

**WS-Discovery** — the ONVIF discovery mechanism. One UDP multicast Probe to
239.255.255.250:3702 asking for NetworkVideoTransmitter; every ONVIF camera
on the segment answers with its service address (XAddrs). This is the part
that removes "know your camera's IP and vendor path" from the install.

**RTSP probe** — a real RTSP handshake (OPTIONS + DESCRIBE) over TCP, so a
failed connection has a NAME. The audit's bar: never "failed to open stream".
The distinct answers, each with its own fix:

    unreachable       nothing answered at host:port (power, cable, IP, VLAN)
    auth              the camera answered 401: wrong username or password
    not-found         connected and authenticated, but the path is wrong (404)
    unsupported-codec stream is up but not H.264/H.265/MJPEG (cv2 can't decode)
    ok                DESCRIBE succeeded; codec parsed from the SDP

cv2's VideoCapture collapses ALL of those into `isOpened() == False`.
"""

from __future__ import annotations

import base64
import re
import socket
import uuid
from urllib.parse import urlparse, urlunparse

from cvti.logging_setup import get_logger

log = get_logger(__name__)

WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)

# WS-Discovery Probe, ONVIF device type. The envelope is fixed boilerplate;
# only the MessageID varies (and must, or caches drop repeat probes).
_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
 xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>uuid:{mid}</w:MessageID>
  <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body>
  <d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>
 </e:Body>
</e:Envelope>"""


def discover(timeout: float = 3.0, *, probe_addr: tuple = WS_DISCOVERY_ADDR) -> list:
    """ONVIF cameras on the local segment: [{ip, xaddr, scopes}], deduped by ip.

    `probe_addr` exists for tests (a unicast responder stands in for the
    multicast group). Failure modes are empty lists, never exceptions — the
    UI's fallback is manual entry either way.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(timeout)
    found: dict = {}
    try:
        sock.sendto(_PROBE.format(mid=uuid.uuid4()).encode(), probe_addr)
        while True:
            try:
                data, (ip, _port) = sock.recvfrom(65535)
            except socket.timeout:
                break
            text = data.decode("utf-8", errors="replace")
            xaddrs = re.search(r"<[^>]*XAddrs[^>]*>([^<]+)<", text)
            scopes = re.search(r"<[^>]*Scopes[^>]*>([^<]+)<", text)
            if not xaddrs:
                continue
            found.setdefault(ip, {
                "ip": ip,
                "xaddr": xaddrs.group(1).strip().split()[0],
                "scopes": (scopes.group(1).strip() if scopes else ""),
                "name": _scope_field(scopes, "name") or _scope_field(scopes, "hardware") or ip,
            })
    except OSError as exc:
        log.warning("ONVIF discovery probe failed: %s", exc)
    finally:
        sock.close()
    log.info("ONVIF discovery: %d camera(s) answered", len(found))
    return list(found.values())


def _scope_field(scopes_match, field: str) -> str:
    if not scopes_match:
        return ""
    m = re.search(rf"onvif://www\.onvif\.org/{field}/([^\s]+)", scopes_match.group(1))
    return m.group(1).replace("%20", " ") if m else ""


# --------------------------------------------------------------------------
# RTSP connection test
# --------------------------------------------------------------------------

_SUPPORTED_CODECS = ("H264", "H265", "HEVC", "JPEG", "MP4V-ES")


def _read_rtsp_response(sock: socket.socket) -> str:
    """One RTSP response: headers always, body when Content-Length says so."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    m = re.search(rb"Content-Length:\s*(\d+)", head, re.I)
    want = int(m.group(1)) if m else 0
    while len(rest) < want:
        chunk = sock.recv(4096)
        if not chunk:
            break
        rest += chunk
    return (head + b"\r\n\r\n" + rest).decode("utf-8", errors="replace")


def probe_rtsp(url: str, timeout: float = 4.0) -> dict:
    """Classify what an RTSP URL actually does. Returns
    {ok, kind, message, codec} — `kind` is one of
    ok | unreachable | auth | not-found | unsupported-codec | not-rtsp.
    Every message says what to DO, not just what happened."""
    u = urlparse(url)
    if u.scheme not in ("rtsp", "rtsps"):
        return {"ok": False, "kind": "not-rtsp",
                "message": f"'{u.scheme or 'this'}' is not an RTSP URL — camera "
                           "streams start with rtsp://", "codec": None}
    host, port = u.hostname, u.port or 554
    # Strip credentials from the request line; send them as Basic auth instead.
    bare = urlunparse((u.scheme, f"{host}:{port}", u.path or "/", u.params, u.query, ""))
    auth = ""
    if u.username:
        token = base64.b64encode(
            f"{u.username}:{u.password or ''}".encode()).decode()
        auth = f"Authorization: Basic {token}\r\n"
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        return {"ok": False, "kind": "unreachable",
                "message": f"Nothing is answering at {host}:{port} ({exc}). Check "
                           "the camera's power, network cable, IP address, and that "
                           "this machine is on the same network/VLAN.", "codec": None}
    try:
        sock.settimeout(timeout)
        sock.sendall((f"DESCRIBE {bare} RTSP/1.0\r\nCSeq: 1\r\n{auth}"
                      "Accept: application/sdp\r\nUser-Agent: Argus\r\n\r\n").encode())
        resp = _read_rtsp_response(sock)
    except OSError as exc:
        return {"ok": False, "kind": "unreachable",
                "message": f"{host}:{port} accepted the connection but did not "
                           f"speak RTSP ({exc}). Is this the camera's RTSP port?",
                "codec": None}
    finally:
        sock.close()

    status = re.match(r"RTSP/\d\.\d\s+(\d+)", resp)
    code = int(status.group(1)) if status else 0
    if code == 401:
        return {"ok": False, "kind": "auth",
                "message": "The camera refused the username or password. Re-enter "
                           "the camera's credentials (often on a sticker on the "
                           "camera, or set in its web page).", "codec": None}
    if code == 404:
        return {"ok": False, "kind": "not-found",
                "message": "Connected and signed in, but this stream path does not "
                           "exist on the camera. Pick your camera's brand for its "
                           "usual path, or check the camera's manual.", "codec": None}
    if code != 200:
        return {"ok": False, "kind": "not-rtsp",
                "message": f"The camera answered RTSP {code or 'nothing'} to "
                           "DESCRIBE — try the brand's stream path, or the ONVIF "
                           "search.", "codec": None}
    codecs = re.findall(r"a=rtpmap:\d+\s+([A-Za-z0-9\-\.]+)/", resp)
    video = [c.upper() for c in codecs]
    good = next((c for c in video if c in _SUPPORTED_CODECS), None)
    if video and not good:
        return {"ok": False, "kind": "unsupported-codec",
                "message": f"The stream is up but uses {'/'.join(video)}, which "
                           "Argus cannot decode. In the camera's settings, switch "
                           "the stream to H.264 (most cameras offer it).",
                "codec": "/".join(video)}
    return {"ok": True, "kind": "ok",
            "message": f"Stream is answering ({good or 'codec unknown'}).",
            "codec": good}
