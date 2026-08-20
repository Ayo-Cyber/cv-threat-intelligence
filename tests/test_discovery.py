"""ONVIF discovery and the RTSP probe (EP-05-T4).

Acceptance pinned here: discovery parses real WS-Discovery answers, and the
connection test gives a DIFFERENT, actionable answer for wrong credentials /
unreachable / wrong path / unsupported codec — never "failed to open stream".

Real camera hardware can't run in CI, so both protocols are spoken by stub
servers built from captured message shapes (Hikvision- and Dahua-style
ProbeMatches; RTSP responses per RFC 2326). The two-brand acceptance check
remains a hardware task for the pilot site.
"""
import socket
import threading
import unittest

from cvti.serving import discovery

HIK_MATCH = """<?xml version="1.0" encoding="UTF-8"?>
<env:Envelope xmlns:env="http://www.w3.org/2003/05/soap-envelope"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
 xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<env:Body><d:ProbeMatches><d:ProbeMatch>
<d:Types>dn:NetworkVideoTransmitter</d:Types>
<d:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/HIKVISION%20DS-2CD2043 onvif://www.onvif.org/hardware/DS-2CD2043</d:Scopes>
<d:XAddrs>http://192.168.1.64/onvif/device_service</d:XAddrs>
</d:ProbeMatch></d:ProbeMatches></env:Body></env:Envelope>"""

DAHUA_MATCH = HIK_MATCH.replace("HIKVISION%20DS-2CD2043", "Dahua%20IPC-HDW2431") \
                       .replace("DS-2CD2043", "IPC-HDW2431") \
                       .replace("192.168.1.64", "192.168.1.108")


class WsDiscoveryTest(unittest.TestCase):
    def _responder(self, payloads):
        """A UDP endpoint that answers any probe with the given payloads."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        addr = sock.getsockname()

        def run():
            data, peer = sock.recvfrom(65535)
            assert b"NetworkVideoTransmitter" in data, "not an ONVIF probe"
            for p in payloads:
                sock.sendto(p.encode(), peer)
            sock.close()

        threading.Thread(target=run, daemon=True).start()
        return addr

    def test_two_brands_of_probe_match_are_parsed(self):
        addr = self._responder([HIK_MATCH, DAHUA_MATCH])
        cams = discovery.discover(timeout=1.5, probe_addr=addr)
        # both answers came from 127.0.0.1, so dedup keeps one record — the
        # fields must still be parsed from the payload
        self.assertEqual(len(cams), 1)
        cam = cams[0]
        self.assertIn("onvif/device_service", cam["xaddr"])
        self.assertIn("HIKVISION", cam["name"])

    def test_a_silent_network_is_an_empty_list_not_a_hang_or_crash(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        try:
            cams = discovery.discover(timeout=0.8, probe_addr=sock.getsockname())
        finally:
            sock.close()
        self.assertEqual(cams, [])


SDP_H264 = ("v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n")
SDP_WEIRD = ("v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 MP2T/90000\r\n")


def _rtsp_server(behavior):
    """One-shot RTSP stub. behavior: '401' | '404' | 'h264' | 'weird' | 'garbage'."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0)); srv.listen(1)

    def run():
        conn, _ = srv.accept()
        conn.recv(4096)
        if behavior == "401":
            resp = ("RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n"
                    'WWW-Authenticate: Basic realm="cam"\r\n\r\n')
        elif behavior == "404":
            resp = "RTSP/1.0 404 Stream Not Found\r\nCSeq: 1\r\n\r\n"
        elif behavior == "garbage":
            resp = "HTTP/1.1 200 OK\r\n\r\nnot rtsp"
        else:
            sdp = SDP_H264 if behavior == "h264" else SDP_WEIRD
            resp = (f"RTSP/1.0 200 OK\r\nCSeq: 1\r\n"
                    f"Content-Length: {len(sdp)}\r\n\r\n{sdp}")
        conn.sendall(resp.encode())
        conn.close(); srv.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


class RtspProbeTest(unittest.TestCase):
    def test_wrong_credentials_are_named_as_credentials(self):
        port = _rtsp_server("401")
        out = discovery.probe_rtsp(f"rtsp://user:wrong@127.0.0.1:{port}/s1")
        self.assertEqual(out["kind"], "auth")
        self.assertIn("username or password", out["message"])

    def test_unreachable_says_unreachable_with_the_host_and_port(self):
        out = discovery.probe_rtsp("rtsp://127.0.0.1:1/s1", timeout=1.0)
        self.assertEqual(out["kind"], "unreachable")
        self.assertIn("127.0.0.1:1", out["message"])
        self.assertIn("power", out["message"])

    def test_a_wrong_path_is_distinct_from_wrong_credentials(self):
        port = _rtsp_server("404")
        out = discovery.probe_rtsp(f"rtsp://127.0.0.1:{port}/wrong")
        self.assertEqual(out["kind"], "not-found")
        self.assertIn("path", out["message"])

    def test_an_unsupported_codec_names_the_codec_and_the_fix(self):
        port = _rtsp_server("weird")
        out = discovery.probe_rtsp(f"rtsp://127.0.0.1:{port}/s1")
        self.assertEqual(out["kind"], "unsupported-codec")
        self.assertIn("MP2T", out["message"])
        self.assertIn("H.264", out["message"])

    def test_a_working_h264_stream_passes_with_its_codec(self):
        port = _rtsp_server("h264")
        out = discovery.probe_rtsp(f"rtsp://127.0.0.1:{port}/s1")
        self.assertTrue(out["ok"])
        self.assertEqual(out["codec"], "H264")

    def test_a_non_rtsp_answer_is_not_reported_as_a_camera_fault(self):
        port = _rtsp_server("garbage")
        out = discovery.probe_rtsp(f"rtsp://127.0.0.1:{port}/s1")
        self.assertFalse(out["ok"])
        self.assertEqual(out["kind"], "not-rtsp")

    def test_credentials_never_appear_in_any_message(self):
        port = _rtsp_server("401")
        out = discovery.probe_rtsp(f"rtsp://admin:hunter2@127.0.0.1:{port}/s1")
        self.assertNotIn("hunter2", out["message"])

    def test_a_non_rtsp_scheme_is_rejected_before_any_network_io(self):
        out = discovery.probe_rtsp("http://127.0.0.1/stream")
        self.assertEqual(out["kind"], "not-rtsp")


if __name__ == "__main__":
    unittest.main()
