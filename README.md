# Argus — CV Threat Intelligence

[![Build CVTI Console (desktop app)](https://github.com/Ayo-Cyber/cv-threat-intelligence/actions/workflows/build-app.yml/badge.svg)](https://github.com/Ayo-Cyber/cv-threat-intelligence/actions/workflows/build-app.yml)

Cheap computer-vision detectors watch your cameras and over-fire constantly. A
vision-language model running **on the same machine** looks at what they flagged
and throws out the ones that aren't real. You get the alerts that matter, and
nothing leaves the building.

On held-out fire footage the raw detectors flagged 90% of ordinary clips as a
threat. With verification: 6.7%, and no fires missed. Every claim, with its
sample size and confidence interval, is in [docs/NUMBERS.md](docs/NUMBERS.md).

---

## What it is

Three layers, one process per site:

| Layer | What it does |
|---|---|
| **Detectors** | YOLO pose/object + a fine-tuned VideoMAE action model + deterministic zone and tamper rules. Fast, per-frame, and deliberately noisy. |
| **Rules engine** | Config-driven, per camera. Decides which detections are the kind of thing *this site* cares about. |
| **Verification gate** | A local VLM (Gemma 3 via Ollama) looks at the frames and confirms or rejects. This is the part that makes the alerts usable. |

Runs offline end to end — no API key, no cloud, no footage leaving the machine.
Measured at 5 concurrent cameras on one laptop.

Full detail: [docs/SYSTEM_GUIDE.md](docs/SYSTEM_GUIDE.md) ·
Design rationale: [docs/architecture.md](docs/architecture.md) ·
Everything else: [docs/README.md](docs/README.md)

---

## Download the desktop app

Installers for macOS, Windows and Linux are on the
[Releases page](https://github.com/Ayo-Cyber/cv-threat-intelligence/releases).

### Verify what you downloaded

Every release lists a SHA-256 for each asset and attaches a `SHA256SUMS.txt`.
Check the file you got is the file we built:

```bash
# macOS / Linux
shasum -a 256 cvti-console-macos.dmg

# Windows (PowerShell)
Get-FileHash cvti-console-windows.zip -Algorithm SHA256
```

Compare against the release body. If it differs, don't run it — tell us.
(Notarised code signing is on the roadmap; until then this is how you confirm
integrity.)

---

## Run it from source

Needs Python 3.9+ and [Ollama](https://ollama.com) for the verification gate.

```bash
git clone https://github.com/Ayo-Cyber/cv-threat-intelligence.git
cd cv-threat-intelligence

python3 -m venv .venv && source .venv/bin/activate    # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

ollama pull gemma3:4b          # the on-device verification model (~3 GB)

./run_demo.sh                  # starts Ollama, the engine and the console together
```

`run_demo.sh` takes an optional site config:

```bash
./run_demo.sh configs/site_theft_demo.json    # 6-camera retail theft wall
./run_demo.sh configs/site_safety_demo.json   # fire / person-down / crowd
```

Ctrl-C in that terminal stops everything.

### The pieces, separately

```bash
# detection engine only — writes confirmed alerts to runs/site/events.db
python -m cvti.serving.pipeline --site-config configs/site_6cam_demo.json \
  --gate-provider ollama --gate-model gemma3:4b --output-dir runs/site

# operator console only, pointed at an existing database
python -m cvti.app.shell --site-config configs/site_6cam_demo.json --db runs/site/events.db

# measure it yourself on the held-out clips
python -m cvti.eval --dataset camnuvem --gate ollama --kind fire
```

> **The gate is not optional.** `--gate-provider mock` confirms *every* alert
> without looking at it, which inverts the product. The engine refuses to start
> on it unless you set `ARGUS_ALLOW_MOCK_GATE=1`, and shows a permanent red
> banner when you do.

### Running the tests

```bash
python -m pytest -q
```

The same suite gates every build — a red suite cannot produce an installer.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'torch'` | Virtualenv isn't active: `source .venv/bin/activate` |
| Engine refuses to start, mentions `ARGUS_ALLOW_MOCK_GATE` | Working as intended — pass `--gate-provider ollama` instead. |
| System panel says "Ollama offline" | `ollama serve`, then Recheck. |
| System panel says "model not pulled" | `ollama pull gemma3:4b` |
| Alerts take ~30s to appear | Expected — that's the local model reasoning about the frames. See the latency notes in [docs/NUMBERS.md](docs/NUMBERS.md). |
| No alerts at all on a demo config | Check `runs/live/run.log`; the engine prints every verdict, confirmed or rejected. |

---

## Where things live

```
cvti/detector/     per-frame detectors (pose, objects, weapons, tamper, situational)
cvti/rules/        config-driven customization engine
cvti/verification/ the VLM gate
cvti/serving/      multi-camera engine, alert queue, sink, notifications
cvti/eval/         the measurement harness — how every number in NUMBERS.md was produced
cvti/app/          operator console (PyQt shell + web UI)
configs/           site configs: cameras, rules, zones per deployment
docs/              see docs/README.md for the index
```

## Data retention

Evidence is deleted **30 days** after recording by default — frames, clips and
the database record together. Configure it in **System → Data retention**.

Two things deliberately outlive their expiry: anything an operator has placed on
**legal hold**, and anything **not yet reviewed** (an open incident must not be
destroyed on a timer while the question is still live). Both are counted in the
System panel, so "why is this still here?" always has an answer.

Argus warns at 85% disk and purges oldest-first at 95%, still refusing to touch
held or unreviewed evidence. Full policy, including erasure and export:
[docs/DATA_RETENTION.md](docs/DATA_RETENTION.md).

## Contributing

`main` is protected; everything lands through a branch and a PR, and CI must be
green. See [CONTRIBUTING.md](CONTRIBUTING.md). Changes worth noticing are
recorded in [CHANGELOG.md](CHANGELOG.md).

## Status

Pre-pilot. The system runs, is measured on three threat classes, and has zero
production deployments. [docs/AUDIT.md](docs/AUDIT.md) is an independent audit of
what's missing; [docs/SPRINT_PLAN.md](docs/SPRINT_PLAN.md) is the plan to close it.
