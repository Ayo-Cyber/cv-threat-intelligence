# Testing Argus on Windows (from a Mac)

*Why this page exists: every bug that actually hurt this project — the
"Ollama doesn't come with it" report, the 189 MB tab switch, sluggishness —
surfaced because a human ran the product on Windows, not because a test did.
Two of those are now covered by CI ([../tests/e2e/](../tests/e2e/)), but a
human still needs a real Windows box for the things no assertion catches:
does it FEEL right, does SmartScreen scare a customer, is the wizard obvious.*

## Option A — a local VM on the Mac (best for repeat testing)

**Apple Silicon** runs Windows 11 **ARM64**; x86 Windows will not boot natively.

1. **UTM** (free, `brew install --cask utm`) — download the *Windows 11 ARM64
   VHDX* via [CrystalFetch](https://github.com/TuringSoftware/CrystalFetch)
   (also free, from Microsoft's own servers), then create a Windows VM in UTM
   pointing at it. Allocate **8 GB RAM / 4 cores / 80 GB disk** — Argus needs
   room for the 3.3 GB model plus evidence.
2. **Parallels Desktop** (paid, ~$100/yr) — smoother, downloads Windows for you,
   better graphics. Worth it if you will test weekly.

**VM caveats that will confuse you if unwarned:**
- **No webcam by default.** UTM/Parallels must be told to share the camera, and
  even then a VM webcam is flaky. Test with **Demo Videos**, or point the VM at
  a real RTSP camera on your LAN — which is the realistic pilot case anyway.
- **CPU-only.** No Metal/CUDA passthrough, so gemma3:4b verdicts take
  noticeably longer than on the Mac. That is a VM artefact, not the product.
- Give it a **bridged network** if you want it to reach LAN cameras.

## Option B — a cloud Windows box (best for a one-off "does the installer work")

An **Azure** or **AWS EC2** Windows Server VM, ~$0.10–0.20/hour, destroyed when
done. Two sizing notes: pick **≥8 GB RAM** (the model), and the first run
downloads 3.3 GB, so a cloud box actually does this *faster* than home
broadband. This is the cheapest honest answer to "does a stranger's machine
run our installer" — and it is a **clean machine**, which is the EP-05
acceptance test we still owe.

## Option C — the free one you already have

**Martins's PC.** He has already installed it once. Give him
[../docs/RUNBOOK.md](RUNBOOK.md) §1 and the release link; his feedback is worth
more than a VM because his machine is a real customer machine.

## The test script (whatever environment you use)

1. Download `argus-windows.zip` from the [latest release](https://github.com/Ayo-Cyber/cv-threat-intelligence/releases/latest); verify against `SHA256SUMS.txt`.
2. Unzip, run `Argus\Argus.exe`. **Expect SmartScreen once** ("More info → Run
   anyway") until the Authenticode certificate exists — note whether that felt
   alarming, because a customer will feel it more.
3. **Create the owner account.** If you are not asked to, you have the wrong
   (pre-v1.0.0) build.
4. Wizard: add a camera (RTSP URL, or skip), **Verification step → the model
   downloads here** (3.3 GB, resumable, progress bar). This is the step that
   reads as "Ollama didn't come with it" — the runtime IS bundled; the weights
   are not. Note how obvious that is, or is not.
5. **Start monitoring.** Watch should show video within ~2 minutes.
6. Switch **FEED → Demo Videos**, confirm alerts appear in Triage with evidence.
7. Type a sentence in **Configure → Rules → Describe it in English**, confirm it
   applies within ~12 s with no restart.
8. Note anything that felt slow, and roughly when — the UI now ships ~70 KB per
   view instead of 189 MB, so sluggishness after this build means something new.

## What CI already proves, so you do not have to check it

- The bundle contains `ollama.exe` and its runner libraries, and the runtime
  executes (`tests/e2e/bundle_runtime_check.py`).
- The shipped `argus-engine.exe` decodes video, publishes authenticated frames
  and persists alerts on a machine with no Python (`tests/e2e/bundle_smoke.py`).
- The full pipeline works against a real RTSP stream, evidence lands on disk,
  unauthenticated frame requests are refused, and a dead verifier produces
  visible UNVERIFIED alerts rather than silence
  (`tests/e2e/docker-compose.yml`).
