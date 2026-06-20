#!/usr/bin/env bash
# Build CVTI.app and wrap it in a distributable .dmg (macOS)
# Usage:  bash scripts/build_mac.sh
set -e

VERSION="0.1.0"
APP_NAME="CVTI"
DMG_NAME="${APP_NAME}-${VERSION}-mac.dmg"
DIST_DIR="dist"

echo "==> Installing build dependencies…"
pip install pyinstaller -q

echo "==> Building ${APP_NAME}.app…"
pyinstaller cvti.spec --clean --noconfirm

APP_PATH="${DIST_DIR}/${APP_NAME}.app"

if [ ! -d "$APP_PATH" ]; then
  echo "ERROR: ${APP_PATH} not found. Check the spec output above."
  exit 1
fi

echo "==> Creating ${DMG_NAME}…"
if command -v create-dmg &>/dev/null; then
  create-dmg \
    --volname "CV Threat Intelligence" \
    --volicon "cvti/app/assets/icon.icns" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 175 190 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 425 190 \
    "${DIST_DIR}/${DMG_NAME}" \
    "${APP_PATH}"
else
  # Fallback: bare hdiutil DMG (no fancy layout)
  hdiutil create \
    -volname "CV Threat Intelligence" \
    -srcfolder "${APP_PATH}" \
    -ov -format UDZO \
    "${DIST_DIR}/${DMG_NAME}"
fi

echo ""
echo "Done: ${DIST_DIR}/${DMG_NAME}"
echo "Send this file — users double-click, drag to Applications, and launch."
