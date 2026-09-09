# Argus Desktop

Electron + React + TypeScript replacement UI, developed alongside the existing
Python/Qt console. The old console and its backend are not replaced or removed.

All UI revamp sources, Electron entry points, adapter, tests and UI documentation
are under `Frontend/` (formerly `desktop/`). The default runtime data directory
remains `runs/desktop/` so existing accounts and evidence continue to load.

## Run on this Mac

From the repository root:

```sh
cd Frontend
npm ci
../.venv/bin/python scripts/prepare_demo.py
npm run desktop
```

The first Electron launch may download the Electron runtime. Node.js 22.12+ is
required by the current locked toolchain (Node 24 LTS is also suitable).

The app starts in **Demo** mode. Four local recordings and two explicitly labelled
sample incidents exercise the interface without invoking detection or a VLM.
Press **Play footage** to play the recordings. Demo settings persist locally and
cannot change the engine's configuration. Media is copied from `data/test_clips`
and excluded from Git; the preparation script does not download videos.

Select **Local engine** inside Electron to sign in to the real backend. The
default account database is separate from the old console: create an owner on
first use. The adapter uses the existing Python virtual environment and:

- Site: `configs/site_live.json`
- Database: `runs/desktop/events.db` (accounts/context in the same run directory)
- Backend: existing `cvti.app.console_backend.ConsoleBackend`

You can select another site and reuse a specific run's accounts/context/events:

```sh
ARGUS_SITE_CONFIG=configs/deluxe_demo.json \
ARGUS_DB=runs/my_desktop_test/events.db \
npm run desktop
```

Paths above are relative to the repository, not `Frontend/`. The real backend
persists camera/rule changes to the selected site. Use a copy of your site config
for experiments. Do not run the old Qt console and this app against the same
camera/engine simultaneously. Previewing cameras is distinct from monitoring.

To run compiled assets rather than Vite:

```sh
npm run build
npm start
```

For a browser-only design preview: `npm run dev`, then open the displayed localhost
URL. Native engine access intentionally does not work in a browser. Browser mode
cannot execute Python or access a webcam through this adapter.

`ARGUS_REPO`, `ARGUS_PYTHON`, and `ARGUS_USER_DATA` optionally override repository,
Python executable and Electron profile paths. On Windows the default virtualenv
path is `.venv/Scripts/python.exe`; Windows native execution is not yet verified.

## What is connected

### Zones and account access

Zones now opens the camera dialog full-screen. The frame and drawing overlay fit
both available width and height without cropping or changing aspect ratio. Drag
a rectangle (default) or select the polygon tool; saved coordinates remain in
original camera pixels. Unsaved drawings prompt before closing or switching tabs.
Drawing requires a readable camera frame, not a running detector. Start monitoring
after saving to test loitering alerts.

Existing installations offer **Forgot password?** and **Create account** at login.
Forgot password displays a Terminal/PowerShell command for the exact selected
database. Run it as the OS user who owns the account database, choose the existing
username, confirm RESET, and enter the new password twice (hidden, 12+ characters).
The command is interactive and never takes passwords as arguments. It preserves
all users, roles, evidence and camera settings; revokes only the selected user's
sessions, clears their lockout, and appends recovery entries to the audit log.
This trusts local OS access, not email verification; protect the computer's OS
account. Recovery cannot be invoked from renderer IPC or the network API.

Default workspace example, from the repository root:

```sh
./.venv/bin/python Frontend/scripts/recover_account.py --db runs/desktop/events.db
```

For a fresh-test workspace use `--db runs/desktop_fresh_test/events.db`. No reset
has been performed on your behalf. Sign in after running the command yourself.

On an existing site, **Create account** directs you to an owner; it does not grant
unauthenticated access. A signed-in owner creates accounts under **Settings >
Users**, choosing Operator, Installer or Owner. Granting Owner requires a second
confirmation. The destructive legacy account-override method remains inaccessible
through the Electron bridge. First-run owner setup is unchanged. Native Windows
verification remains outstanding.

- Real login/first-owner setup and existing backend authorization.
- Camera wall and engine start/stop, with separate preview and monitoring states.
- Existing webcam/demo/live-stream feed registry and switch progress.
- Camera discovery/testing/add/remove and site area creation/assignment.
- Per-camera scene evidence, editable context, approval and remapping.
- Existing site/area review, with bulk approval blocked on missing evidence/conflicts.
- Detector switches/presets, English watch rules, scanner status.
- Polygon/rectangle zones in original camera coordinates, dwell thresholds and removal.
- Incident evidence playback, acknowledgement, real/false/inconclusive review and notes.
- Six-step site setup, readiness checks, notification tests and retention settings.
- Light/dark themes, responsive layouts and keyboard-accessible dialogs.

Backend errors are surfaced; unavailable operations never return invented success.
The UI does not change the detection algorithms or their existing limitations.
See [migration status](docs/MIGRATION_STATUS.md) for remaining feature parity.

## Verify

```sh
npm test
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
npm run build
../.venv/bin/python scripts/smoke_backend.py --repo "$(cd .. && pwd)"
ARGUS_REPO="$(cd .. && pwd)" node scripts/smoke_electron.mjs
```

With Vite running in a second terminal and Google Chrome installed:

```sh
npm run test:ui
```

Native smoke tests create temporary site/account data and remove it afterward.
They do not start monitoring or alter the real site. Browser tests include real
media readiness, responsive overflow, scene draft preservation, persistence,
rule switches, zoning, incident review and separation from native engine access.

## Architecture

```text
React renderer / typed transport
  -> sandboxed Electron preload (one narrow invoke interface)
  -> Electron main process (sender and command validation)
  -> Python JSON-lines worker (authentication and permission checks)
  -> existing ConsoleBackend -> existing scene/detection/verification services
```

Renderer Node integration is disabled. Context isolation and sandboxing are on.
Arbitrary IPC commands and navigation are rejected. Credentials stay in the
backend session, not localStorage. Demo localStorage is separate sample data.
Camera images use the backend's token-protected localhost MJPEG service.
The UI makes no external font requests. The configured backend verifier may still
use whichever provider the existing installation selects.

As of Ayo's `main` at `7900c69`, the FastAPI read/auth/WebSocket skeleton exists,
including stream descriptors and go2rtc support. Configuration/write endpoints
are intentionally absent there. This UI still uses the existing Python worker
for its complete read/write workflows; it does not yet consume that API or its
WebRTC descriptors. The transport boundary can be migrated as those contracts mature.
This is a source-run desktop application, not a signed macOS/Windows installer.

## See the live EarthCams

1. Launch the native app with `npm start` and select **Local engine**.
2. Sign in (or create the first owner for the selected run directory).
3. Open **Overview**, then choose **Live EarthCams** under **Camera source**.
4. Wait for source resolution to finish. The existing backend resolves fresh
   stream URLs using yt-dlp; internet access is required.
5. Press the circular **Connect camera feeds** button beside **All areas** on
   the camera wall. This starts preview without starting detection.
6. Choose **Start monitoring** separately when you want inference and alerts.

The current registry lists Dublin Street, LaGuardia Airport, Venice Canal and
Heathrow Airport. These are external streams, not guaranteed to remain online.
The separate **Demo** workspace intentionally does not connect to them.
If a source cannot resolve, the UI displays the backend error rather than
substituting a recording. Do not run the old console against these same cameras
at the same time during testing.
