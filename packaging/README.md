# Packaging — CVTI Console desktop app

Bundles Argus into a standalone installation with three executables sharing one
PyInstaller `COLLECT`: the Qt operator console (`Argus`), detection pipeline
(`argus-engine`), and Electron control-plane backend (`argus-api`). No Python
install is needed on the target machine.

## What's in the bundle (and what isn't)

**In:** the UI and API, PyQt6/WebEngine, YOLO/Ultralytics, VideoMAE, SigLIP
object-enrollment code, OpenAI CLIP (including tokenizer BPE data), ONNX
Runtime, OpenCV, and SQLite. Build only from a clean checkout with controlled
release inputs: the spec deliberately includes top-level `configs/*.json` and
`configs/*.yaml`, plus optional demo/model directories when they exist, so
`.gitignore` alone does not prevent local or private files entering an artifact.

**Out (on purpose):** the TrueSight verifier model and SigLIP model weights.
The SigLIP Python runtime is bundled, but its runtime configuration must point
to a locally provisioned model directory; missing weights report
`unavailable` and are not auto-downloaded. Customer databases and private
configuration must be kept outside the clean build inputs described above.

## Build locally on macOS arm64

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r packaging/requirements-build.txt
env -u PYTHONPATH -u PYTHONHOME .venv/bin/python -m PyInstaller \
  packaging/argus.spec --noconfirm --clean --distpath dist --workpath build
```

Artifacts:
| OS      | Output                                   |
|---------|------------------------------------------|
| macOS   | `dist/Argus.app`                         |
| Windows | `dist/Argus/Argus.exe`                   |
| Linux   | `dist/Argus/Argus`                       |

> **PyInstaller does not cross-compile.** The local frozen-build validation is
> for macOS arm64. Windows and Linux use the CI matrix and remain pending until
> their packaged smoke reports pass; a successful Mac build does not validate
> either platform.

## Frozen object-enrollment validation

After building, use provisioned SigLIP weights and a real reference photograph:

```bash
.venv/bin/python tests/e2e/frozen_api_object_watch.py \
  --bundle dist/Argus --model /absolute/path/to/siglip-base-patch16-224 \
  --image /absolute/path/to/reference.jpg \
  --require-frozen --assert-no-source-imports
```

The harness relocates the bundle, uses disposable accounts and data, and checks
missing-model readiness, reviewed enrollment, a real embedding, activation and
restart persistence. Import checks reject developer-source dependencies while
explicitly checking PyTorch's deterministic generated RemoteModule source.
The configurable `--startup-timeout` defaults to 180 seconds, matching Electron.

The September 18, 2026 macOS arm64 run passed: one 768-dimensional finite,
unit-normalized embedding, persisted activation, and both import reports checked.
Initial startup took 115.597 seconds; restart took 0.634 seconds. This is not a
startup-performance guarantee, recognition-accuracy result, signed-installer test,
or validation of packaged live proposals, TrueSight, notifications or other OSes.

## Legacy Qt macOS bundle and DMG

`packaging/make_dmg.sh` packages the legacy Qt `dist/Argus.app`. The production
desktop release is the Electron installer, which wraps the three-executable
payload around `dist/Argus`; use the CI/electron-builder release path for that
artifact rather than presenting this DMG as the production installer.

```bash
python packaging/build.py --clean --dmg     # build the .app AND the .dmg
# or, from an existing dist/Argus.app:
bash packaging/make_dmg.sh
```

Produces `dist/Argus.dmg` with the standard drag-to-Applications layout.
Recipient mounts it and drags **Argus** into **Applications**.

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
"dist/Argus.app/Contents/MacOS/Argus" \
  --site-config configs/site_video_demo.json --db runs/site_vlm/events.db
```
