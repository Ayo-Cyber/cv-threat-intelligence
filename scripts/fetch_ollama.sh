#!/usr/bin/env bash
# Fetch the Ollama binary into vendor/ollama/<platform>/ so cvti.spec can bundle it.
# Run this once before `pyinstaller cvti.spec`. The model itself is NOT downloaded
# here — it pulls on the app's first run. (~120 MB runtime, not the ~3.3 GB model.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    PLAT="darwin"
    ASSET="ollama-darwin.tgz"
    ;;
  Linux)
    PLAT="linux"
    case "$ARCH" in
      x86_64|amd64) ASSET="ollama-linux-amd64.tar.zst" ;;
      aarch64|arm64) ASSET="ollama-linux-arm64.tar.zst" ;;
      *) echo "Unsupported Linux arch: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Use scripts/fetch_ollama.bat on Windows." >&2; exit 1
    ;;
esac

DEST="$REPO_ROOT/vendor/ollama/$PLAT"

# Already fetched? Skip unless FORCE=1 (the runtime is ~430 MB — don't re-pull).
if [[ -f "$DEST/ollama" && "${FORCE:-0}" != "1" ]]; then
  echo "Ollama already present at $DEST/ollama (set FORCE=1 to re-download)."
  exit 0
fi

URL="https://github.com/ollama/ollama/releases/latest/download/$ASSET"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $ASSET ..."
curl -fL "$URL" -o "$TMP/$ASSET"

echo "Extracting ..."
mkdir -p "$DEST"
case "$ASSET" in
  *.tgz)      tar -xzf "$TMP/$ASSET" -C "$DEST" ;;
  *.tar.zst)  tar --zstd -xf "$TMP/$ASSET" -C "$DEST" ;;
esac

# Normalize: ensure an executable named `ollama` sits at the top of $DEST
# (some archives place it under bin/).
if [[ ! -f "$DEST/ollama" && -f "$DEST/bin/ollama" ]]; then
  cp "$DEST/bin/ollama" "$DEST/ollama"
fi
chmod +x "$DEST/ollama" 2>/dev/null || true

if [[ -f "$DEST/ollama" ]]; then
  echo "Done: $DEST/ollama"
else
  echo "Extraction finished but no 'ollama' binary found in $DEST — inspect it manually." >&2
  exit 1
fi
