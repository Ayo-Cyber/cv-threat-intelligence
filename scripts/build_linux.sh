#!/usr/bin/env bash
# Build CVTI as a self-contained Linux AppImage
# Usage:  bash scripts/build_linux.sh
set -e

VERSION="0.1.0"
APP_NAME="CVTI"
DIST_DIR="dist"

echo "==> Installing build dependencies…"
pip install pyinstaller -q

echo "==> Fetching Ollama binary for the offline VLM…"
bash "$(dirname "$0")/fetch_ollama.sh"

echo "==> Building binary bundle…"
pyinstaller cvti.spec --clean --noconfirm

BUNDLE_DIR="${DIST_DIR}/${APP_NAME}"

if [ ! -d "$BUNDLE_DIR" ]; then
  echo "ERROR: ${BUNDLE_DIR} not found."
  exit 1
fi

# ---- AppImage (optional — requires appimagetool) ----
if command -v appimagetool &>/dev/null; then
  echo "==> Packaging as AppImage…"
  APPDIR="${DIST_DIR}/AppDir"
  mkdir -p "${APPDIR}/usr/bin" "${APPDIR}/usr/lib"

  cp -r "${BUNDLE_DIR}/." "${APPDIR}/usr/bin/"

  cat > "${APPDIR}/cvti.desktop" <<EOF
[Desktop Entry]
Name=CV Threat Intelligence
Exec=CVTI
Icon=cvti
Type=Application
Categories=Utility;
EOF

  # Placeholder icon — replace with real one
  touch "${APPDIR}/cvti.png"

  ARCH=$(uname -m) appimagetool "${APPDIR}" "${DIST_DIR}/${APP_NAME}-${VERSION}-${ARCH}.AppImage"
  echo "Done: ${DIST_DIR}/${APP_NAME}-${VERSION}-*.AppImage"
else
  # Plain tarball fallback
  TAR="${DIST_DIR}/${APP_NAME}-${VERSION}-linux-x86_64.tar.gz"
  tar czf "${TAR}" -C "${DIST_DIR}" "${APP_NAME}/"
  echo "Done: ${TAR}"
  echo "Users extract and run ./${APP_NAME}/CVTI"
fi
