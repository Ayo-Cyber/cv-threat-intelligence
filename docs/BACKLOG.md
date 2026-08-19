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

## EP-07-T3 — Measure `weapons` and `violence` · *blocked on data*

`baseline_weapon` and `baseline_violence` are `critical` priority in
`configs/baseline_critical_v1.json` — always on, in every customer config — and
`docs/NUMBERS.md` lists both as *"built and demonstrable, not yet validated"*.
The two detectors that would summon an armed response have never been measured.

**Blocked because:** the acceptance asks for ≥50 clips per detector. There are 6
matching clips in `data/test_clips/`. The plan names RWF-2000 (violence) and a
weapons benchmark, both of which need acquiring and a licence review — that is
the blocker, not compute.

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
