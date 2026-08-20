"""The incident record as a PDF (EP-06-T2).

This is the product's actual deliverable made tangible: what a manager reviews,
and what a customer hands to an insurer or the police. It must therefore be a
file that leaves the building intact — not a screen.

Written by hand rather than via a PDF library, deliberately. The record is text
plus JPEG evidence frames, and JPEG is the one image format PDF embeds verbatim
(`DCTDecode`) — so the whole job is a few text objects and image XObjects. A
dependency would buy generality this file does not need, and every dependency
has to build inside a PyInstaller bundle on three platforms.

The layout is plain on purpose: A4, one metadata block, the model's reasoning,
the human's conclusion, then one evidence frame per block. An incident record
is evidence, and evidence should look like a document, not a brochure.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

PAGE_W, PAGE_H = 595, 842            # A4 in points
MARGIN = 50
LINE = 14


# Base-font PDF text is latin-1; map the unicode punctuation we actually use
# rather than letting it degrade to '?'.
_PUNCT = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
          "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u2192": "->",
          "\u00b7": "-", "\u2026": "..."}


def _esc(text: str) -> str:
    for uni, ascii_ in _PUNCT.items():
        text = text.replace(uni, ascii_)
    return (text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            .encode("latin-1", "replace").decode("latin-1"))


def _wrap(text: str, width: int = 92) -> list:
    out = []
    for raw_line in (text or "").splitlines() or [""]:
        words, line = raw_line.split(), ""
        if not words:
            out.append("")
        for word in words:
            if len(line) + len(word) + 1 > width:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            out.append(line)
    return out


def _jpeg_size(data: bytes) -> tuple:
    """(width, height) from JPEG SOF markers; (0, 0) if unparseable."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return (int.from_bytes(data[i + 7:i + 9], "big"),
                    int.from_bytes(data[i + 5:i + 7], "big"))
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return (0, 0)


class _Pdf:
    """Just enough PDF: pages of text lines and embedded JPEGs."""

    def __init__(self):
        self.objects: list = []          # 1-indexed by position+1
        self.pages: list = []            # (content_obj, [(name, img_obj)])

    def _add(self, body: bytes) -> int:
        self.objects.append(body)
        return len(self.objects)

    def add_jpeg(self, data: bytes) -> tuple:
        w, h = _jpeg_size(data)
        obj = self._add(
            (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
             f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
             f"/Length {len(data)} >>\nstream\n").encode() + data + b"\nendstream")
        return obj, w, h

    def add_page(self, stream: bytes, images: list) -> None:
        compressed = zlib.compress(stream)
        content = self._add(
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode()
            + compressed + b"\nendstream")
        self.pages.append((content, images))

    def render(self) -> bytes:
        n_fixed = len(self.objects)
        page_ids = [n_fixed + 3 + i for i in range(len(self.pages))]
        catalog_id, pages_id = n_fixed + 1, n_fixed + 2
        font = "/F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> " \
               "/F2 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

        tail = [f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode(),
                (f"<< /Type /Pages /Count {len(self.pages)} /Kids "
                 f"[{' '.join(f'{p} 0 R' for p in page_ids)}] >>").encode()]
        for content, images in self.pages:
            xo = " ".join(f"/Im{obj} {obj} 0 R" for obj, _, _ in images)
            tail.append(
                (f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                 f"/Resources << /Font << {font} >> /XObject << {xo} >> >> "
                 f"/Contents {content} 0 R >>").encode())

        every = self.objects + tail
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(every, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(every) + 1}\n0000000000 65535 f \n".encode()
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(every) + 1} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref_at}\n%%EOF\n").encode()
        return bytes(out)


class _PageWriter:
    """Cursor-based text/image layout that starts new pages as it fills."""

    def __init__(self, pdf: _Pdf):
        self.pdf = pdf
        self.ops: list = []
        self.images: list = []
        self.y = PAGE_H - MARGIN

    def _flush(self):
        if self.ops or self.images:
            self.pdf.add_page("\n".join(self.ops).encode("latin-1"), self.images)
        self.ops, self.images, self.y = [], [], PAGE_H - MARGIN

    def need(self, height: float):
        if self.y - height < MARGIN:
            self._flush()

    def text(self, line: str, *, size: int = 10, bold: bool = False,
             colour: str = "0 0 0", gap: float = LINE):
        self.need(gap)
        font = "F2" if bold else "F1"
        self.ops.append(f"BT /{font} {size} Tf {colour} rg "
                        f"{MARGIN} {self.y:.0f} Td ({_esc(line)}) Tj ET")
        self.y -= gap

    def rule(self):
        self.need(10)
        self.ops.append(f"0.75 0.78 0.82 RG 0.5 w {MARGIN} {self.y:.0f} m "
                        f"{PAGE_W - MARGIN} {self.y:.0f} l S")
        self.y -= 12

    def jpeg(self, data: bytes, caption: str):
        obj, w, h = self.pdf.add_jpeg(data)
        if not w or not h:
            self.text(f"[unreadable frame: {caption}]", colour="0.6 0.2 0.2")
            return
        max_w, max_h = PAGE_W - 2 * MARGIN, 300
        scale = min(max_w / w, max_h / h, 1.0)
        dw, dh = w * scale, h * scale
        self.need(dh + LINE + 8)
        self.y -= dh
        self.ops.append(f"q {dw:.0f} 0 0 {dh:.0f} {MARGIN} {self.y:.0f} cm "
                        f"/Im{obj} Do Q")
        self.images.append((obj, w, h))
        self.y -= 4
        self.text(caption, size=8, colour="0.45 0.48 0.55", gap=LINE + 4)

    def done(self):
        self._flush()


def build_incident_pdf(event: dict, frames: list, dest: str | Path) -> Path:
    """`event` is the row as a dict; `frames` is [(caption, jpeg_bytes), ...]."""
    pdf = _Pdf()
    page = _PageWriter(pdf)

    page.text("ARGUS — INCIDENT RECORD", size=16, bold=True, gap=22)
    page.text(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} — evidence record; "
              f"handle under the site's data-protection terms.",
              size=8, colour="0.45 0.48 0.55", gap=18)
    page.rule()

    def field(label, value):
        page.text(f"{label}:  {value if value not in (None, '') else '—'}", size=10)

    field("Incident", f"#{event.get('id')} — {event.get('rule')}")
    field("Camera", event.get("camera_id"))
    field("Time", event.get("iso"))
    field("Priority", str(event.get("priority", "")).upper())
    conf = event.get("confidence")
    field("Model confidence", f"{float(conf):.2f}" if conf is not None else None)
    if event.get("unverified"):
        page.text("UNVERIFIED — TrueSight could not decide; reviewed manually.",
                  bold=True, colour="0.55 0.35 0.05")
    page.rule()

    page.text("Why the system raised this", bold=True, gap=16)
    for line in _wrap(event.get("reason") or "(no model reasoning recorded)"):
        page.text(line)
    page.y -= 6

    page.text("Human conclusion", bold=True, gap=16)
    owner = event.get("owner")
    outcome = event.get("outcome")
    state = event.get("state") or "new"
    if state == "resolved":
        field("Responded by", owner)
        field("Conclusion", {"real": "REAL incident", "false_alarm": "False alarm",
                             "inconclusive": "Inconclusive"}.get(outcome, outcome))
        field("Resolved at", time.strftime("%Y-%m-%d %H:%M:%S",
              time.localtime(event["resolved_at"])) if event.get("resolved_at") else None)
    elif state == "acknowledged":
        page.text(f"OPEN — claimed by {owner or 'unknown'}, not yet concluded.",
                  bold=True, colour="0.55 0.35 0.05")
    else:
        page.text("OPEN — nobody has responded to this incident yet.",
                  bold=True, colour="0.55 0.35 0.05")
    for line in _wrap(event.get("note") or ""):
        page.text(f"Note: {line}" if line == _wrap(event.get("note"))[0] else line)
    page.rule()

    page.text(f"Evidence — {len(frames)} frame(s)", bold=True, gap=16)
    if not frames:
        page.text("(no evidence frames on disk — they may have expired under the "
                  "retention policy)", colour="0.45 0.48 0.55")
    for caption, data in frames:
        page.jpeg(data, caption)

    page.done()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf.render())
    log.info("incident record written: %s (%d frame(s))", dest, len(frames))
    return dest
