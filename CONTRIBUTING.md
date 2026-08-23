# Contributing

## Branches and pull requests

`main` is protected and always releasable. Nothing lands on it directly —
everything goes through a branch and a pull request, including one-line fixes
and documentation.

What GitHub enforces on `main`:

| Rule | Setting |
|---|---|
| Direct pushes | blocked — a PR is required |
| Required check | **Test suite** must pass |
| Branch freshness | must be up to date with `main` before merging |
| Approvals required | 0 — a solo maintainer can merge their own PR once CI is green |
| Force pushes / deletion | blocked |
| Conversation resolution | required before merge |

Admins are *not* subject to these rules, deliberately: it leaves one escape
hatch for a genuine emergency. Using it should feel like a decision, not a
shortcut.

```bash
git checkout main && git pull
git checkout -b <type>/<short-description>
# ... work, commit ...
git push -u origin <type>/<short-description>
gh pr create --fill
```

Branch names use the same prefixes as commits:

| Prefix | For |
|---|---|
| `feat/` | new capability |
| `fix/` | a defect |
| `perf/` | measured speed or memory work |
| `docs/` | documentation only |
| `ci/` | build, release, workflow |
| `chore/` | housekeeping with no behaviour change |

**A PR merges when CI is green.** The test job gates the installer build, so a
red suite cannot produce a release. Don't merge around it.

## Commits

Conventional-commit prefix, then a subject that says what changed *for the
user*, not which file moved:

```
fix(gate): the mock gate confirms everything — refuse to start on it silently
```

The body explains **why**, and what would have gone wrong without the change.
This repo's history is a large part of its documentation — a reader six months
out should be able to reconstruct the reasoning without asking anyone.

Quote measured numbers with the sample size they came from. `docs/NUMBERS.md` is
generated from archived eval runs (`python tools/make_numbers_sheet.py`) and must
never be hand-edited.

## Before you open a PR

```bash
python -m pytest -q
```

If your change alters behaviour, it needs a test that fails without it. Tests
here encode policy, not just mechanics — `test_never_sheds_the_last_camera` is
the shape to aim for.

Update `CHANGELOG.md` under `## [Unreleased]` for anything a user or operator
would notice. Skip it for pure internal refactors.

## Things that are deliberately hard to do by accident

- **The mock gate confirms every alert without looking at it.** Engines refuse to
  start on it unless `ARGUS_ALLOW_MOCK_GATE=1` is set, and show a permanent red
  banner when they do. If a change makes that guard easier to bypass, it is the
  wrong change.
- **Prompt text is a measured surface.** The wording in `gate.py`
  (`_QUESTIONS`, `_DETECTOR_QUESTIONS`, the templates) determines the headline
  precision figure — three revisions moved it 26 points. Change any of it and CI
  fails until you re-measure:

  ```bash
  ollama serve &
  python tools/prompt_regression.py run                     # compare to baseline
  python tools/prompt_regression.py run --update-baseline   # accept the new numbers
  ```

  and commit `docs/prompt_baseline.json`. CI does not re-measure — a GitHub
  runner cannot run a 3 GB VLM — it enforces that the wording and the recorded
  measurement agree. If the golden set is missing locally, rebuild it with
  `python tools/prompt_regression.py capture` (slow, once).
- **Generated files are generated.** `docs/NUMBERS.md` comes from
  `tools/make_numbers_sheet.py`; `reports/` is eval output and is not tracked.
- **Large binaries stay out of git.** The nine demo clips in `data/test_clips/`
  and `models/weapon_best.pt` are tracked deliberately and listed as explicit
  exceptions in `.gitignore`. Anything else large belongs elsewhere.

## Releasing

Tag `main` once CI is green:

```bash
git tag v0.10.0 && git push origin v0.10.0
```

That builds macOS, Windows and Linux installers, publishes them to a GitHub
Release with SHA-256 sums, and only runs if the test job passed.
