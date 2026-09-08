"""latency_baseline.py — measure what the wall actually costs, before replacing it.

    python tools/latency_baseline.py                 # synthetic camera, full path
    python tools/latency_baseline.py --seconds 30
    python tools/latency_baseline.py --json
    python tools/latency_baseline.py --source rtsp://user:pw@cam/stream1

W1 proposes go2rtc + WebRTC against a <300ms glass-to-glass SLO, on the theory
that the MJPEG wall is slow. Nobody has ever measured the MJPEG wall. This is
the yardstick: it runs the REAL production path — StreamDecoder with the
low-latency capture flags, the smooth-publish loop with its viewer boost, the
FramePublisher's multipart HTTP — and reports where the milliseconds sit, so
the go2rtc work is scored against a number instead of an adjective.

HOW IT MEASURES
---------------
A synthetic camera (this process + ffmpeg, H.264 ultrafast/zerolatency over
mpegts/tcp — the encode a real camera does) burns the send time into every
frame as a grid of large black/white blocks: 48 bits of ms-since-epoch plus an
8-bit checksum, drawn fat enough to survive H.264 and JPEG q70. Both ends are
this machine, so one clock serves both. Two probes then read that clock back:

  camera-to-engine   peek_latest() on the decoder — when detection could first
                     see the frame. Camera encode + transport + decode. This
                     leg is paid REGARDLESS of the wall transport: detection
                     needs decoded frames. WebRTC cannot remove it.
  glass-to-glass     the frame arriving at a real MJPEG client (the same
                     multipart stream the wall tiles read), JPEG-decoded.

  wall overhead      the difference: publish pacing + JPEG encode + HTTP.
                     THIS is the slice W1's WebRTC path can actually win back
                     (plus the client's render, which neither transport
                     escapes). If it is small, W1's case is codec reach and
                     robustness, not speed — better to know before the build.

With --source (a real rtsp:// camera) there is no burned clock, so only the
wall overhead is measured (publisher receive → client receive, one clock, this
machine); the camera leg is reported as not measurable in software.

Boundaries stated plainly: the client is a Python reader, not a browser, so
browser JPEG-decode/paint (~one frame) is outside the number on both sides of
any comparison. The synthetic camera's x264 encode stands in for the real
camera's encoder — same job, same place in the path.

Needs ffmpeg on PATH for the synthetic source (brew install ffmpeg /
apt install ffmpeg). Runs ~25s by default and prints a W0-style table; the
decoder's own stage timings land in perf_report.json beside it, readable with
tools/perf_readout.py.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# the burned clock: 48 bits of ms-since-epoch + 8-bit XOR checksum, 2 rows of
# 28 blocks. Blocks are 36px with the value averaged over their inner core, so
# H.264 ringing and JPEG blocking at the edges cannot flip a bit.
# --------------------------------------------------------------------------

CLOCK_BITS = 48
CHECK_BITS = 8
TOTAL_BITS = CLOCK_BITS + CHECK_BITS          # 56 = 2 rows x 28 cols
CLOCK_COLS = 28
CLOCK_ROWS = 2
BLOCK_PX = 36
CLOCK_MARGIN = 24                              # grid offset from frame corner
CLOCK_BORDER = 12                              # solid white surround

FRAME_W, FRAME_H = 1280, 720
# 15fps, like the pilot's cameras. Must stay BELOW the publisher's 24fps viewer
# boost: over tcp nothing ever drops, so a source faster than the decoder turns
# the deficit into ever-growing queue latency — the rig measuring itself.
SOURCE_FPS = 15.0


def _checksum(value: int) -> int:
    """XOR of the six data bytes — one flipped block fails the frame."""
    c = 0
    for i in range(CLOCK_BITS // 8):
        c ^= (value >> (8 * i)) & 0xFF
    return c


def encode_clock(frame, ms: int) -> None:
    """Burn `ms` into the frame's top-left corner, in place."""
    import numpy as np
    word = ((ms & (1 << CLOCK_BITS) - 1) << CHECK_BITS) | _checksum(ms)
    x0, y0 = CLOCK_MARGIN, CLOCK_MARGIN
    w = CLOCK_COLS * BLOCK_PX + 2 * CLOCK_BORDER
    h = CLOCK_ROWS * BLOCK_PX + 2 * CLOCK_BORDER
    frame[y0:y0 + h, x0:x0 + w] = 255                      # the white surround
    for i in range(TOTAL_BITS):
        row, col = divmod(i, CLOCK_COLS)
        bit = (word >> (TOTAL_BITS - 1 - i)) & 1
        y = y0 + CLOCK_BORDER + row * BLOCK_PX
        x = x0 + CLOCK_BORDER + col * BLOCK_PX
        frame[y:y + BLOCK_PX, x:x + BLOCK_PX] = 255 if bit else 0
    _ = np                                                  # numpy is required upstream


def decode_clock(frame) -> int | None:
    """The ms burned into the frame, or None when compression mangled it.

    Scale-aware: the publisher resizes the wall to max_width=640, so the same
    grid arrives at the client at half size. Geometry is linear in width, and
    an 18px block still carries a clean bit through H.264 + JPEG q70."""
    scale = frame.shape[1] / float(FRAME_W)
    if scale < 0.2:                                        # nothing readable left
        return None
    px = BLOCK_PX * scale
    origin = (CLOCK_MARGIN + CLOCK_BORDER) * scale
    core = px / 5.0                                        # ignore block edges
    word = 0
    for i in range(TOTAL_BITS):
        row, col = divmod(i, CLOCK_COLS)
        y, x = origin + row * px, origin + col * px
        block = frame[int(y + core):int(y + px - core),
                      int(x + core):int(x + px - core)]
        if block.size == 0:
            return None
        word = (word << 1) | (1 if float(block.mean()) > 127.0 else 0)
    ms = word >> CHECK_BITS
    return ms if (word & 0xFF) == _checksum(ms) else None


# --------------------------------------------------------------------------
# the synthetic camera: frames piped to ffmpeg, H.264 out over mpegts/tcp.
# --------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SyntheticCamera:
    """A live H.264 network stream whose every frame says when it was born.

    ffmpeg listens on tcp and the engine's decoder connects to it — so the
    decoder's reconnect logic covers the startup race for free, the same way
    the e2e compose leans on it.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self.url = f"tcp://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "SyntheticCamera":
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{FRAME_W}x{FRAME_H}", "-r", str(int(SOURCE_FPS)),
               "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
               "-x264-params", "keyint=25:min-keyint=25:scenecut=0", "-bf", "0",
               # mpegts muxdelay defaults to 0.7 SECONDS — without these two,
               # the harness reports ffmpeg's own buffer as camera latency
               # (measured: a rock-steady 806ms p50 that vanished with them).
               "-muxdelay", "0", "-muxpreload", "0", "-flush_packets", "1",
               "-f", "mpegts", f"tcp://127.0.0.1:{self.port}?listen=1"]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._feed, name="synthetic-camera",
                                        daemon=True)
        self._thread.start()
        return self

    def _feed(self) -> None:
        import numpy as np
        period = 1.0 / SOURCE_FPS
        # A textured backdrop, not flat grey: x264 on a flat frame is
        # unrealistically cheap and fast. Noise once, reused — the clock grid
        # is the only thing that changes per frame, which also keeps the
        # generator itself off the measured path.
        rng = np.random.default_rng(7)
        backdrop = rng.integers(40, 200, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        next_at = time.perf_counter()
        while not self._stop.is_set():
            frame = backdrop.copy()
            encode_clock(frame, int(time.time() * 1000))   # birth stamp, last thing
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                return                                      # ffmpeg gone; run over
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# --------------------------------------------------------------------------
# the two probes
# --------------------------------------------------------------------------

def watch_decoder(decoder, samples: list, stop: threading.Event) -> None:
    """Probe (a): the moment detection could first see each frame."""
    last_seq = -1
    while not stop.is_set():
        try:
            frame, seq = decoder.peek_latest()
        except Exception:  # noqa: BLE001 - a probe must never wobble the run
            frame, seq = None, last_seq
        if frame is not None and seq != last_seq:
            last_seq = seq
            ms = decode_clock(frame.image)
            if ms is not None:
                samples.append(time.time() * 1000.0 - ms)
        time.sleep(0.002)


def read_mjpeg(url: str, on_jpeg, stop: threading.Event) -> None:
    """Probe (b): a real client of the same multipart stream the wall reads.

    Raw socket on purpose — urllib buffers, and a buffered reader measuring
    latency reports its own buffer.
    """
    from urllib.parse import urlparse
    u = urlparse(url)
    sock = socket.create_connection((u.hostname, u.port), timeout=10)
    try:
        sock.sendall((f"GET {u.path}?{u.query} HTTP/1.1\r\n"
                      f"Host: {u.hostname}\r\nConnection: close\r\n\r\n").encode())
        buf = b""
        while not stop.is_set():
            chunk = sock.recv(65536)
            if not chunk:
                return
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")              # JPEG SOI
                if start < 0:
                    buf = buf[-2:]
                    break
                end = buf.find(b"\xff\xd9", start + 2)     # JPEG EOI
                if end < 0:
                    if start > 0:
                        buf = buf[start:]
                    break
                on_jpeg(buf[start:end + 2], time.time() * 1000.0)
                buf = buf[end + 2:]
    finally:
        sock.close()


def _stats(values: list) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    pick = lambda f: ordered[min(len(ordered) - 1, int(round(f * (len(ordered) - 1))))]  # noqa: E731
    return {"count": len(ordered), "p50_ms": round(pick(0.50), 1),
            "p95_ms": round(pick(0.95), 1), "max_ms": round(ordered[-1], 1),
            "mean_ms": round(sum(ordered) / len(ordered), 1)}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run(seconds: float, source: str | None, out_dir: Path, as_json: bool) -> int:
    import cv2

    synthetic = source is None
    if synthetic and not _ffmpeg_available():
        print("ffmpeg not found on PATH — needed for the synthetic camera.\n"
              "  brew install ffmpeg   |   apt install ffmpeg\n"
              "Or point --source at a real rtsp:// camera (wall overhead only).",
              file=sys.stderr)
        return 2

    camera = SyntheticCamera(_free_port()).start() if synthetic else None
    url = camera.url if synthetic else source

    from cvti.serving.frame_publisher import FramePublisher
    from cvti.serving.pipeline import MultiStreamPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    publisher = FramePublisher().start(out_dir)
    # The REAL wall path, production defaults: one view-only camera, so frames
    # take the glass route and never enter detection — but the model, the
    # smooth-publish loop, and the viewer boost all behave exactly as shipped.
    pipe = MultiStreamPipeline({"glass": url}, publisher=publisher,
                               view_only={"glass"})
    pipe.start()

    stop = threading.Event()
    decode_lat: list = []           # (a) camera -> engine
    glass_lat: list = []            # (b) camera -> wall client
    recv_times: list = []           # client-side arrival clock, any source

    threading.Thread(target=watch_decoder, args=(pipe._decoders["glass"],
                                                 decode_lat, stop), daemon=True).start()

    def on_jpeg(jpeg: bytes, at_ms: float) -> None:
        recv_times.append(at_ms)
        if not synthetic:
            return
        import numpy as np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        ms = decode_clock(img)
        if ms is not None:
            glass_lat.append(at_ms - ms)

    stream_url = (f"http://127.0.0.1:{publisher.port}/stream/glass"
                  f"?token={publisher.token}")
    reader = threading.Thread(target=read_mjpeg, args=(stream_url, on_jpeg, stop),
                              daemon=True)
    reader.start()

    warmup = 5.0
    print(f"measuring the production MJPEG wall path for {seconds:.0f}s "
          f"(+{warmup:.0f}s warm-up) — source: "
          f"{'synthetic camera (H.264 ultrafast, mpegts/tcp)' if synthetic else url}",
          file=sys.stderr)
    time.sleep(warmup)
    decode_lat.clear(), glass_lat.clear(), recv_times.clear()
    time.sleep(seconds)
    stop.set()
    time.sleep(0.3)

    # The decoder's own stage numbers rode along on the BOARD the whole time —
    # leave them where the readout expects them.
    from cvti.serving.perf import write_report
    report_path = write_report(out_dir)

    camera and camera.stop()
    publisher.stop()
    for d in pipe._decoders.values():
        d.stop()

    fps = 0.0
    if len(recv_times) > 1 and recv_times[-1] > recv_times[0]:
        fps = (len(recv_times) - 1) / ((recv_times[-1] - recv_times[0]) / 1000.0)

    a, b = _stats(decode_lat), _stats(glass_lat)
    doc = {
        "source": "synthetic" if synthetic else url,
        "transport": "mjpeg (multipart/x-mixed-replace)",
        "camera_to_engine_ms": a,
        "glass_to_glass_ms": b,
        "wall_overhead_p50_ms": (round(b["p50_ms"] - a["p50_ms"], 1)
                                 if a.get("count") and b.get("count") else None),
        "client_fps": round(fps, 1),
        "perf_report": str(report_path) if report_path else None,
        "slo_ms": 300,
    }

    if as_json:
        print(json.dumps(doc, indent=2))
        return 0

    print()
    print("MJPEG WALL BASELINE — the number W1 must beat")
    print(f"  received at client: {fps:.1f} fps")
    if a.get("count"):
        print(f"  camera -> engine    p50 {a['p50_ms']:7.1f} ms   p95 {a['p95_ms']:7.1f} ms"
              f"   (n={a['count']})  <- detection pays this regardless of transport")
    if b.get("count"):
        against = " MEETS the 300ms SLO" if b["p95_ms"] < 300 else \
                  "  exceeds the 300ms SLO"
        print(f"  glass -> glass      p50 {b['p50_ms']:7.1f} ms   p95 {b['p95_ms']:7.1f} ms"
              f"   (n={b['count']}) {against}")
    if doc["wall_overhead_p50_ms"] is not None:
        print(f"  wall overhead       p50 {doc['wall_overhead_p50_ms']:7.1f} ms"
              f"                         <- the slice WebRTC can win back")
    elif not synthetic:
        print("  glass -> glass      not measurable on a real camera in software "
              "(no clock in the frames)")
    if b.get("count") == 0 and synthetic:
        print("  ! no clock frames decoded at the client — stream never came up? "
              "check ffmpeg, and rerun with a longer --seconds")
    if report_path:
        print(f"  stage detail:       python tools/perf_readout.py {out_dir}")
    return 0 if (not synthetic or b.get("count")) else 1


def _ffmpeg_available() -> bool:
    from shutil import which
    return which("ffmpeg") is not None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure the current MJPEG wall's glass-to-glass latency.")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="measurement window after the 5s warm-up (default 20)")
    ap.add_argument("--source", default=None,
                    help="a real rtsp:// URL instead of the synthetic camera "
                         "(wall overhead only — real frames carry no clock)")
    ap.add_argument("--out-dir", default="runs/latency_baseline",
                    help="where frames.json / perf_report.json land")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return run(args.seconds, args.source, Path(args.out_dir), args.json)


if __name__ == "__main__":
    raise SystemExit(main())
