#!/usr/bin/env bash
# Package dist/CVTI Console.app into a distributable .dmg (macOS only).
#
#   bash packaging/make_dmg.sh
#
# Produces dist/CVTI-Console.dmg — mount it, drag the app to Applications.
# Uses hdiutil (built into macOS), so no extra tooling required.
set -euo pipefail

APP_NAME="CVTI Console"
VOL_NAME="CVTI Console"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist"
APP="$DIST/$APP_NAME.app"
DMG="$DIST/CVTI-Console.dmg"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "make_dmg.sh only runs on macOS." >&2; exit 1
fi
if [[ ! -d "$APP" ]]; then
  echo "Missing $APP — run 'python packaging/build.py' first." >&2; exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Lay out the mount contents: the app + a shortcut to /Applications so the
# user can drag-drop to install.
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder "$STAGE" \
  -ov -format UDZO \
  "$DMG" >/dev/null

echo "built: $DMG"
du -sh "$DMG" | awk '{print "size: "$1}'
