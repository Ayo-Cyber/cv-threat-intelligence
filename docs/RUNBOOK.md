# Argus — Pilot Runbook

*Deploy, diagnose, recover. Written so someone who is not the author can run a
pilot site. Companion: [ARCHITECTURE.md](../ARCHITECTURE.md) for how it works,
[SECURITY.md](../SECURITY.md) for the security model.*

## 1 · Deploy

**From the installer (the customer path):**
1. Run the platform installer (`Argus.dmg` / Windows zip from the Releases
   page — verify the SHA-256 against `SHA256SUMS.txt` on the release).
2. Launch Argus → create the owner account (nothing ships with a password).
3. The wizard walks: add cameras (ONVIF search or IP + vendor path) → verify
   feed → draw zones → pick a use-case template → confirm detectors →
   verification (downloads the ~3.3 GB TrueSight model once, resumable) →
   send a test alert → self-test names anything missing, in plain English.
4. **Start monitoring.** Models load ~2 min; the footer flips to
   "N of N cameras live".

**From source (the dev path):**
```
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt
python -m pytest -q          # green before you trust the checkout
python -m cvti.app.shell     # (or the `argus` shell alias)
```

**Per-site configuration lives in one JSON** (the site file). Notify channel,
retention days and heartbeat apply **live** (within ~5 s); cameras, zones and
detector toggles apply at the next Start monitoring — the UI says which.

## 2 · Diagnose

Order of trust: **the product's own surfaces first** — they were built to make
silence impossible.

| Symptom | Look at | Meaning / fix |
|---|---|---|
| "Nothing is being detected" | Watch top bar (engine truth line) | "Engine not running" → Start monitoring. Error counts → gate trouble (below). |
| A camera tile dashed, `NO SIGNAL` | Link row under the tile | `OFFLINE 6m · N attempts` = the camera or its network, not Argus. Power/cable/VLAN, then *Test* on the Cameras screen — failures are named (wrong credentials / unreachable / wrong path / unsupported codec). |
| Alerts arrive UNVERIFIED (dashed grey) | Settings → Local AI gate | Ollama down or model missing. The wizard's Verification step starts/downloads it. UNVERIFIED means "shown anyway, unjudged" — by design. |
| No alerts on the phone | `/health` → `heartbeat`/`alert_latency`; mobile row in health reasons | Mobile serves on 8710 (walks to 8714 if busy); total failure is named in `/health`. Telegram creds are env vars on the engine machine. |
| Suspected data problem | Settings → Audit trail | Chain says "intact" or "treat as tampered" — believe it. |
| Anything else | Settings → Diagnostics → Download diagnostics | Logs + health snapshot, **no footage**. Attach to the issue. |

**Files that tell the truth** (site output dir, default `runs/site/`):
`gate_health.json` (fresh = engine alive; `reasons[]` names every degradation
including "configured detector's model failed to load"), `monitor.log`
(engine stdout), and per-component logs in `~/Library/Application
Support/Argus/logs/` (macOS) / `%APPDATA%\Argus\logs` (Windows).

## 3 · Recover

**Target recovery time: 30 minutes from bare replacement hardware to
monitoring, given the installer and a config backup.** (Measured path: install
≤10 min + model download on site bandwidth + restore ≤1 min + camera checks.)

### 3.1 Config restore (disk died, machine replaced)
1. Install Argus, create an owner account (accounts are per-install security
   material — they are deliberately NOT in backups).
2. Settings → Backup & restore → **Restore** from the newest
   `argus-config-*.zip` (daily, automatic, kept 14 deep — from the external
   drive/NAS if `backup_dir` was set, else the old machine's
   `Application Support/Argus/backups/`). Cameras, zones, rules (including
   plain-English rules), detector settings and routing all return.
3. Start monitoring. Verify each camera's link row goes `live`.

A backup zip is plain: `unzip -l argus-config-*.zip` works even if Argus
doesn't. `site.json` + `configs/zones/*` + `configs/rules/*` + routing.

### 3.2 Corrupt events.db
Automatic: at every app start the store is integrity-checked; a corrupt file
is **quarantined intact** as `events.corrupt-<stamp>.db` beside a fresh store,
and the failure is loud. To attempt salvage:
`sqlite3 events.corrupt-*.db ".recover" | sqlite3 events.db` — then restart.

### 3.3 Evidence recovery
If `evidence backup` was pointed at a NAS (`backup_evidence`), event folders
and a consistent `events.db` snapshot are there. Without it, evidence on a
dead disk is gone — that is the documented trade; retention would have
deleted most of it anyway.

### 3.4 Roll back a bad update
Settings → Updates → previous version, or offline:
`python -m cvti.updates rollback` — works with no network, including back to
the as-shipped version. Updates are Ed25519-verified; an unsigned archive
never installs.

## 4 · Keys, credentials, access — the bus-factor inventory

| Thing | Where | Recovery |
|---|---|---|
| Site owner account | per-install `auth.db` (hashed) | No recovery by design. Reset = delete `auth.db` on the box (physical access = first-run screen). All other users re-created by the owner. |
| Update-channel signing key (Ed25519) | `~/.argus-release.key` on the release machine + **must be** in the company password manager | Lost = cannot ship updates; re-key requires a new release embedding the new public key. Public half: `cvti/updates.py`. |
| Apple / Windows signing certs | Developer portals + password manager (see [SIGNING.md](SIGNING.md)) | Revoke + re-issue from the portals; CI picks up new secrets. |
| GitHub repo + CI secrets | `Ayo-Cyber` account, secrets under repo Settings | Listed name-by-name in SIGNING.md. |
| Telegram/WhatsApp notifier creds | env vars on the site machine | Re-issue from the bot provider; nothing stored in the repo. |
| Heartbeat receiver key | site file (`heartbeat_key`) + receiver config | Rotate both ends; see [HEARTBEAT.md](HEARTBEAT.md). |

## 4.5 · Memory policy for the verifier

The bundled Ollama is started with `OLLAMA_NUM_PARALLEL=2`,
`OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_KV_CACHE_TYPE=q8_0` — without these,
Ollama's defaults (4 parallel slots × full-context KV caches + the vision
tower) grow the 3.3 GB gemma3:4b to **~13 GB resident**. With them it runs at
roughly a third of that; extra concurrent verifications queue server-side.
**If you run `ollama serve` yourself** (brew service, the Ollama app), Argus
uses your server as-is — set the same variables in its environment to get the
same footprint. The model unloads automatically after ~5 minutes idle;
`ollama stop gemma3:4b` frees it immediately.

## 5 · Routine operations

- **Weekly owner summary**: automatic, Mondays 08:00, via the site notifier;
  PDFs accumulate in `<output>/summaries/`. On demand: Value → Weekly summary.
- **Daily assurance**: the engine self-tests a real frame through the real
  gate daily and says "all systems normal" out loud — silence is never the
  success signal. Absence of the daily message IS a signal: check §2.
- **Backups**: automatic daily on app start; point `backup_dir` at an external
  drive/NAS in Settings for off-box copies. Evidence backup is opt-in.
