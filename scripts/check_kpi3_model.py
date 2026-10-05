"""Verify the configured production encoder using a local image, without enrollment."""
import argparse
import json
import math
import time
from pathlib import Path

from cvti.object_watch.runtime_config import resolve_config, load_configured_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    backend = load_configured_backend(resolve_config(args.root))
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vector = backend.embed_image(args.image.read_bytes())
    report = {
        "backend": backend.name,
        "fingerprint": backend.fingerprint,
        "dimensions": len(vector),
        "vector_norm": math.sqrt(sum(v * v for v in vector)),
        "load_seconds": load_seconds,
        "embedding_seconds": time.perf_counter() - started,
        "recognition_accuracy_tested": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
