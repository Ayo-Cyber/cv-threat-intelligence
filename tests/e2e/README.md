# End-to-end tests

Until 25 Aug, CI proved the bundle *imported* (`argus-engine --help`). Every
bug that actually hurt — the Windows runtime question, per-feed identity
fragmentation, a 189 MB alert payload — was found by a human running the
product. These tests close that gap.

## The container pipeline (`docker-compose.yml`)

```
docker compose -f tests/e2e/docker-compose.yml up --build \
  --abort-on-container-exit --exit-code-from assert
```

A real **mediamtx** RTSP server, a real **ffmpeg** camera looping a clip into
it, and the **real engine** consuming it — no mocks in the data path. Then
`assert_pipeline.py` checks ten things, each one a failure this project has
actually had:

| # | Assertion | The failure it forbids |
|---|---|---|
| 1 | the engine writes `/health` | a fully-broken site that says nothing |
| 2 | the RTSP camera reaches CONNECTED | decode/reconnect silently never working |
| 3 | health is *fresh* | the UI trusting a 20-hour-old snapshot |
| 4 | frames publish **with a token** | the live wall dark for days (it could not authenticate to its own frames) |
| 5 | anonymous frame requests are refused | evidence on an open port |
| 6 | alerts reach `events.db` | pixels in, nothing out |
| 7 | a dead verifier ⇒ **UNVERIFIED** alerts | a transport failure read as "TrueSight said it's safe" |
| 8 | evidence frames land on disk | alerts with nothing to review |
| 9 | the phone view is up and login-gated | the mobile view unreachable, or worse, open |
| 10 | every schema column the app reads exists | a migration that silently did not run |

## The shipped-bundle tests (run in CI on all three OSes)

- **`bundle_smoke.py`** runs the *packaged* `argus-engine` against real video on
  a machine with no Python: does it decode, publish authenticated frames, and
  persist alerts with the right schema?
- **`bundle_runtime_check.py`** (Windows) proves `ollama.exe` **and its runner
  libraries actually shipped inside the bundle** and execute — the question
  "does Ollama come with it?" that nothing in CI could previously answer.

## Notes for whoever changes these

- The RTSP image is minimal: **no shell, no `nc`, no `curl`**. Do not add a
  `healthcheck` that assumes one — that is how this compose file failed on its
  first run. Let the camera retry and the engine reconnect; that path deserves
  exercising anyway.
- The gate points at a **dead port on purpose**. CI must not need a 3.3 GB
  model, and "the verifier is unreachable" is the case most worth asserting.
- Assertions wait with deadlines rather than sleeping a fixed time — a slow
  runner should be slow, not flaky.
- **The engine must outlive the assertions.** `--abort-on-container-exit` tears
  the stack down when *any* container exits, so if the engine's `--seconds`
  budget expires first it kills the assertions mid-run — exit 137, which looks
  like a failure and is not. The engine runs 900s; the assertions finish in
  ~90s and their exit code is the verdict.
