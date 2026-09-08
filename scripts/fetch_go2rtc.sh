#!/usr/bin/env bash
# Fetch the go2rtc binary into vendor/go2rtc/<platform>/ so packaging/argus.spec
# can bundle it (W1: the stream gateway). One static binary, ~9 MB — pinned to a
# version on purpose: a silent major bump in a stream gateway is a field
# incident waiting for a pilot. Override with GO2RTC_VERSION=... FORCE=1.
set -euo pipefail

VERSION="${GO2RTC_VERSION:-v1.9.14}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    PLAT="darwin"
    case "$ARCH" in
      arm64)  ASSET="go2rtc_mac_arm64.zip" ;;
      x86_64) ASSET="go2rtc_mac_amd64.zip" ;;
      *) echo "Unsupported macOS arch: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  Linux)
    PLAT="linux"
    case "$ARCH" in
      x86_64|amd64)  ASSET="go2rtc_linux_amd64" ;;
      aarch64|arm64) ASSET="go2rtc_linux_arm64" ;;
      *) echo "Unsupported Linux arch: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Use scripts/fetch_go2rtc.bat on Windows." >&2; exit 1
    ;;
esac

DEST="$REPO_ROOT/vendor/go2rtc/$PLAT"

if [[ -f "$DEST/go2rtc" && "${FORCE:-0}" != "1" ]]; then
  echo "go2rtc already present at $DEST/go2rtc (set FORCE=1 to re-download)."
  exit 0
fi

URL="https://github.com/AlexxIT/go2rtc/releases/download/$VERSION/$ASSET"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading go2rtc $VERSION ($ASSET) ..."
curl -fL "$URL" -o "$TMP/$ASSET"

mkdir -p "$DEST"
case "$ASSET" in
  *.zip)
    unzip -oq "$TMP/$ASSET" -d "$TMP/x"
    # the zip holds one binary, whatever it is named
    mv "$(find "$TMP/x" -type f | head -1)" "$DEST/go2rtc"
    ;;
  *)
    mv "$TMP/$ASSET" "$DEST/go2rtc"
    ;;
esac
chmod +x "$DEST/go2rtc"
echo "go2rtc $VERSION -> $DEST/go2rtc"
