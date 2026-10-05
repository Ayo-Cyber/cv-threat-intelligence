"""One embedding through a locally installed SigLIP: prove the files work.

    python -m cvti.object_watch.smoke <model_dir>

Prints one JSON line: {"ok": true, "fingerprint": ..., "dimensions": 768,
"seconds": ...} or {"ok": false, "error": ...}. Exit code follows `ok`.
Called by the installer after download; the install is only marked ready
when this passes. Structural checks (files present) are not a successful load.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def check(model_dir: Path) -> dict:
    started = time.time()
    try:
        import io
        from PIL import Image, ImageDraw
        from cvti.object_watch.embeddings import SiglipEmbeddingBackend
        backend = SiglipEmbeddingBackend(str(model_dir))
        canvas = Image.new("RGB", (224, 224), (245, 245, 245))
        ImageDraw.Draw(canvas).rectangle((64, 64, 160, 160), fill=(200, 40, 40))  # anything deterministic
        buf = io.BytesIO(); canvas.save(buf, format="PNG")
        vector = backend.embed_image(buf.getvalue())
        dims = int(len(vector))
        if dims <= 0:
            return {"ok": False, "error": "empty embedding"}
        return {"ok": True, "fingerprint": getattr(backend, "fingerprint", None),
                "dimensions": dims, "seconds": round(time.time() - started, 2)}
    except Exception as exc:  # noqa: BLE001 - SILENT-OK: the reason IS the output; the
        # installer logs it and shows it to the operator as the install failure.
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(json.dumps({"ok": False, "error": "usage: smoke <model_dir>"}))
        return 2
    out = check(Path(argv[0]))
    print(json.dumps(out))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
