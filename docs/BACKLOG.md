# Backlog

Work that is started, blocked, or deliberately deferred. Each entry says what
state it is in and what would unblock it, so nothing is carried in someone's
head. Live plan: [SPRINT_PLAN.md](SPRINT_PLAN.md) · findings: [AUDIT.md](AUDIT.md).

---

## EP-07-T1 — Prompt regression suite · *code written, not measured*

**Branch:** `feat/ep-07-t1-prompt-regression` (draft PR) · **Status:** parked 19 Aug 2026

Built and working:

- `cvti/eval/golden.py` — freezes the detector stage as a labelled corpus of
  real gate inputs, so a prompt change can be measured without re-running YOLO
  and VideoMAE. 96 candidates captured from 36 CamNuvem clips.
- `cvti/eval/prompt_fingerprint.py` — hashes only the text that reaches the
  model, so refactors don't trip it. Covers all five prompt tables.
- `tools/prompt_regression.py` — `capture` / `run` / `check`.
- CI step that fails when the prompt wording changes without a re-measurement.

**Why it is parked:** the baseline measurement needs ~96 calls to the local VLM,
roughly 20 minutes with `gemma3:4b` pinning ~4 GB. That is a deliberate,
scheduled job, not something to run mid-session.

**To finish:**

1. Make the replay resumable — write per-case results to a `.jsonl` as it goes,
   the way `cvti/eval/harness.py` already does. It currently writes nothing until
   it completes, so an interrupted run loses everything.
2. Add `--limit` so the routine run is a couple of minutes rather than twenty.
3. Run `python tools/prompt_regression.py run --update-baseline` and commit
   `docs/prompt_baseline.json`. Three tests in `tests/test_prompt_regression.py`
   fail until that exists — by design; the guard is meant to be unsatisfiable
   until someone measures.

---

## EP-07-T3 — Measure `weapons` and `violence` · *harness ready, blocked on data only*

The two critical always-on detectors are still unmeasured, but the machinery
now exists end-to-end (built 20 Aug, smoke-verified through the real models):
the moment clips land, measuring is one command per detector.

**Getting the clips** (licence question for legal: research-licensed sets used
for internal benchmarking of a commercial product — clips never ship, never
train, results published as aggregate rates only):

- **Violence — downloadable today:** UCF-Crime (crcv.ucf.edu) — Fighting +
  Assault categories clear the 50-clip floor; Normal supplies negatives.
  Drop the official layout under `data/ucf_crime/`.
- **Weapons:** UCF-Crime `Shooting` + the Seville mock-attack CCTV set
  (deepknowledge-us.github.io) + **a staged self-capture session** (one
  afternoon, prop weapon, CCTV-height camera — full rights, pilot-matched
  angles). Curated clips go under `data/critical/weapons/{threat,normal}/`.
- RWF-2000 needs SMIIP Lab approval for commercial use — one email, don't wait.

**Then:**

    python tools/measure_critical.py status        # inventory vs the >=50 floor
    python tools/measure_critical.py run violence  # needs ollama up (~clips × ~12s)
    python tools/measure_critical.py run weapons

The tool refuses to publish below 50 threat / 20 normal clips (--smoke for
wiring checks), archives to runs/eval/<kind>-v1/ in the standard format, and
prints the exact DETECTOR_VALIDATION + NUMBERS.md lines — which tests then
hold consistent. Note: the eval harness previously never loaded the weapon
model at all (the flag was set, no model passed) — an eval would have scored
0% recall against a detector that wasn't running. Fixed and pinned by test.

---

## EP-01-T1 — logging in a real PyInstaller bundle · *verified only by simulation*

The acceptance asks to verify explicitly that logging works inside the packaged
build, "path resolution differs". It was verified by monkeypatching `sys.frozen`
and `sys._MEIPASS` in a test — **no bundle was ever built**. The frozen build is
precisely where logs are otherwise unretrievable, so simulated is not verified.

**To finish:** run `python packaging/build.py --clean`, launch the bundle, and
confirm logs land in the per-user application-support directory.

---

## EP-00-T6 — release checksums · *never executed*

The SHA-256 step only runs on a `v*` tag, and no tag has been pushed since it
landed. The shell was tested locally; the CI path is unproven. It will be
exercised by the next real release.

---

## EP-02-T1 — legal paperwork · *with the legal team*

Privacy policy, DPIA, ToS, DPA template, professional indemnity insurance.
External; gates the pilot absolutely. Not engineering work.

## W6 — KPI evaluation data still to source · *manifest live, three rows SMOKE*

**Status:** first real-gate scorecard run 9 Sep 2026 (committed at
`runs/eval/kpi/scorecard.md`); rows below their n floors and the sources
earmarked to fill them:

- **person / intrusion / loitering** (21/20/16 of 73): VIRAT Ground 2.0
  (event annotations — the honest filler; big staged download, check disk),
  PETS 2007 loitering scenarios, more fetcher passes. Kaggle
  `ibrahimarif31/ucf-crime-with-fence-climbing` (4 GB) for intrusion —
  Ayo said keep in backlog, don't pull yet (9 Sep).
- **DCSASS categories already on disk but unmapped** (`data/staging/dcsass`):
  Robbery/Burglary/Vandalism segments could deepen row 9 for free.

**Unblocks:** disk headroom + one word from Ayo.

## W6 — kipshidze shoplifting set is invisible to Stage 1 · *needs eyeballs*

**Status:** found by the first real-gate run. 92 shoplifting clips; only 9%
ever produced a detector CANDIDATE — not gate strictness, the
concealment/pose stage never fires on that footage style. Twenty minutes of
watching a handful of clips decides whether it is a detector blind spot
(serious, W3/W5-adjacent) or a data quirk (staging, angle, thefts past the
30s analysis cap). Until then the row's number carries the set honestly.

## Kipshidze Stage-1 blindness: open-vocab is NOT the fix (measured 10 Sep)
Probe: YOLO-World with concealment phrases ("person putting an item into a
bag" etc.) over the 92 kip theft clips + 90 kip normals, 5 frames each
(runs/eval/attr/kip_openvocab_probe.json). Result: 100% theft candidates but
79-88% of NORMALS fire too — a static phrase matches "person near bag", not
the act. The discriminating signal is temporal; that is video-action's job,
and its kip blindness is a training-domain gap. The theft lever remains the
video-action retrain on domain data (GPU hours, cloud) — not more W3.
Also worth raising with the customer: kip is staged eye-level phone-style
footage; whether it belongs in the SIGNED theft row is a definition question
(real shop CCTV is ceiling-mounted, where Stage-1 already proposes).
