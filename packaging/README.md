# Packaging — CVTI Console desktop app

Bundles the operator app (`cvti/app/shell.py`, a Qt WebEngine window over the
web UI) into a standalone, double-clickable app per OS. No Python install needed
on the target machine.

## What's in the bundle (and what isn't)

**In:** the UI, the camera-onboarding logic, the events viewer, PyQt6 +
WebEngine, OpenCV (for live snapshots), SQLite.

**Out (on purpose):** the detection engine — YOLO / VideoMAE / the VLM gate.
That's the heavy `cvti.serving.pipeline` component (torch, ultralytics,
transformers, + the Ollama VLM). It runs as a **separate service** on the edge
box, writes `events.db`, and the app reads it. Keeping it out is why the app
bundle is ~250 MB (Qt) instead of several GB.

## Build for your current OS

```bash
pip install pyinstaller PyQt6 PyQt6-WebEngine opencv-python-headless numpy
python packaging/build.py --clean
```

Artifacts:
| OS      | Output                                   |
|---------|------------------------------------------|
| macOS   | `dist/CVTI Console.app`                  |
| Windows | `dist/CVTI Console/CVTI Console.exe`     |
| Linux   | `dist/CVTI Console/CVTI Console`         |

> **PyInstaller does not cross-compile.** A Windows `.exe` must be built on
> Windows, a Linux binary on Linux. Build each on its own machine, or use CI.

## macOS: build a .dmg installer

```bash
python packaging/build.py --clean --dmg     # build the .app AND the .dmg
# or, from an existing dist/CVTI Console.app:
bash packaging/make_dmg.sh
```

Produces `dist/CVTI-Console.dmg` (~230 MB) with the standard drag-to-Applications
layout. Recipient mounts it and drags **CVTI Console** into **Applications**.

> ⚠️ **The app is not code-signed or notarized.** On the Mac it was built on it
> runs fine. On *another* Mac (downloaded/AirDropped), Gatekeeper will quarantine
> it — the user sees "unidentified developer" or "damaged and can't be opened".
> Workarounds: right-click → **Open** (first launch only), or
> `xattr -dr com.apple.quarantine "/Applications/CVTI Console.app"`.
> For real distribution you need an **Apple Developer ID** certificate +
> `codesign` + `notarytool` (a ~$99/yr account). That's a separate task.

## All three OSes from one push (CI)

`.github/workflows/build-app.yml` runs a matrix on `macos-latest`,
`windows-latest`, and `ubuntu-latest`. Trigger it from the **Actions** tab
(`workflow_dispatch`) or by pushing a `v*` tag; each job uploads a zipped
artifact you can download.

## Pointing the app at a site / events DB

The bundled app defaults to `configs/site_live.json` and `runs/site/events.db`
relative to its working directory. To point elsewhere, launch the inner binary
with args (also the way to see log output, since the GUI build is windowed):

```bash
# macOS — run the executable inside the .app to see stdout
"dist/CVTI Console.app/Contents/MacOS/CVTI Console" \
  --site-config configs/site_video_demo.json --db runs/site_vlm/events.db
```
