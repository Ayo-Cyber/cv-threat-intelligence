# Project Context

## Project Name
AI-Powered Threat Detection for Camera Systems

## Current Stage
Very early-stage proof of concept (POC).

The immediate goal is not to build a production platform yet. The first objective is to prove that a camera feed can be connected to a computer vision model and that the model can detect a small set of threat-related behaviors in a live demo.

## Founding Team
- `Demilade`: AI/ML Engineer, Co-founder
- Friend / technical partner: Founder, Software Engineer

## Team Reality
- This is a founder-led project.
- The AI/ML side will need strong practical guidance and iteration.
- The software engineering side will help with system integration, backend, productization, and scaling after the initial POC is validated.

## High-Level Vision
Build an AI-powered surveillance intelligence layer that can sit on top of existing camera systems and detect threats or suspicious activities in real time.

The long-term vision is to support real-world security use cases relevant to Nigeria and similar markets, especially scenarios that may not be well covered by generic public datasets.

## Why This Exists
Traditional camera systems are mostly passive: they record footage, but a human usually has to watch or review incidents manually.

This project aims to make camera systems proactive by:
- monitoring video feeds automatically
- detecting suspicious or dangerous events
- producing alerts quickly
- creating a foundation for later deployment into real camera environments

## Immediate Goal
Establish a working POC.

The POC should show that:
- a live camera feed can be connected to a model
- the model can process frames in near real time
- the system can detect selected threat scenarios during a controlled demo
- the system can visibly report or alert when a threat is detected

## POC Strategy
Start simple and controlled before thinking about deployment at scale.

Planned first setup:
- use a webcam for initial testing
- later test with a rented IP camera
- keep `RTSP` compatibility in mind from the beginning, since most deployment-grade cameras expose RTSP streams

The webcam-based setup is mainly for fast iteration. The rented camera phase is for validating that the same approach can work with more realistic camera infrastructure.

## What The First POC Should Demonstrate
The founders want to stage a live demo in front of a camera and see whether the system can detect selected actions such as:
- violent behavior or fighting
- attempted theft
- visible weapons such as knives or guns

This initial demo is intended to answer one main question:

"Can we build a working live threat-detection pipeline that is believable enough to justify deeper investment and scaling?"

## Important Product Direction
The long-term product should not assume a webcam-only environment.

It should be designed with future support for:
- `RTSP` camera ingestion
- real CCTV or IP camera deployments
- later model retraining and tuning on locally relevant data
- eventual edge or on-prem deployment options

## Nigeria-Specific Opportunity
The team believes Nigeria has important security scenarios that are underrepresented in generic datasets and off-the-shelf models.

Examples already identified:
- warehouse workers stealing from owners
- motorcycle bag snatching in commercial areas
- other locally specific edge cases to be defined later

Because of this, the project may eventually require custom data collection and model fine-tuning for local conditions, behaviors, camera angles, clothing styles, traffic patterns, and scene context.

## Planned Data Strategy
The current understanding is:
- first prove the concept works with an initial model and a controlled live demo
- later hire or arrange actors to reenact important threat scenarios
- record those scenarios as custom training data
- use that data to improve accuracy for locally relevant use cases

This means the data strategy is expected to evolve in two phases:

### Phase 1
Use existing models, public datasets, and a controlled webcam/camera demo to establish feasibility.

### Phase 2
Collect custom Nigeria-relevant data and fine-tune models for stronger real-world performance.

## Technical Assumptions Right Now
These are current assumptions, not final decisions:
- use computer vision for live video threat detection
- start with a webcam
- later connect to an `RTSP` camera
- train or fine-tune a model for selected threat categories
- evaluate the model live during staged scenarios

## Deployment Concerns Already Identified
Deployment is a major concern, especially hardware and GPU requirements.

There has already been advice from another ML engineer mentioning:
- deployment complexity on camera/edge systems
- `NVIDIA Jetson`-style deployment thinking
- possibly using a virtual machine or environment that mimics Jetson for early experimentation
- rough cost ideas around `$200` for an early setup option

These notes are still exploratory and not yet confirmed. The exact deployment approach is still undefined and should be treated as an open question rather than a settled plan.

## Clarification On POC vs Production
The first POC does **not** need to solve final deployment.

Right now, the priority is:
1. show the detection pipeline works
2. measure basic performance
3. identify what fails
4. learn what kind of data and hardware will be needed next

Only after that should the team decide:
- whether to use cloud GPU, local GPU, or edge hardware
- whether Jetson is actually needed for the next phase
- whether model optimization is necessary immediately

## Working Assumptions For The First Iteration
For now, the most practical first iteration is likely:
- webcam input
- one local machine
- one initial model
- a small set of demo threat classes
- visible on-screen detections and simple alerting

That would be enough to establish a credible POC before dealing with:
- multi-camera scaling
- RTSP fleet support
- dashboards
- mobile alerts
- edge deployment
- customer pilots

## Recommended Initial POC Scope
Keep the first version narrow.

Suggested first scope:
- connect a webcam feed
- run inference frame by frame
- detect people and selected threat-related actions or objects
- display bounding boxes / labels / confidence scores
- optionally raise a simple on-screen or console alert when a threat class is detected
- save short clips or frames from detections for review

## Candidate POC Threat Categories
These are candidate classes for the earliest experiments:
- person
- knife
- gun
- fighting / violence
- theft-like action

Important note:
- `knife` and `gun` may be easier to approach earlier as object-detection problems
- `fighting` and `theft` are more behavior/action-recognition problems and may be harder for a first pass

## Practical Guidance Principle
Because the project is still early and the founder team is learning while building, decisions should favor:
- speed of learning over completeness
- simple demos over complex architecture
- measurable experiments over assumptions
- narrow scope over broad promises

## What Success Looks Like For This POC
The POC can be considered successful if the team can demonstrate:
- a live video feed connected to a model
- at least some meaningful threat detections in a controlled scenario
- acceptable responsiveness for a demo
- enough evidence that the idea is worth refining

It does **not** need to be perfect, production-grade, or fully deployable yet.

## Open Questions
These questions are still unresolved and should guide next planning:
- Which exact model family should be used first?
- Should the first version focus on object detection, action recognition, or a hybrid pipeline?
- What threat categories are realistic for a first milestone?
- What hardware is available right now?
- Will training happen from scratch, or will the team fine-tune an existing model?
- How will the POC be evaluated: qualitative demo only, or also with metrics?
- What is the shortest path from webcam demo to RTSP camera demo?
- Is Jetson emulation actually useful at this stage, or is a normal local GPU/CPU setup enough for the first experiment?

## Current Priority
The current priority is to establish the first POC, learn from it quickly, and only then make clearer decisions about scaling, custom data collection, and deployment architecture.

## Checkpoint 2026-04-01
The project has now moved beyond a purely planning stage.

Current repo state:
- `MVP_POC_PLAN.md` exists and defines a 36-hour POC milestone
- `detector.py` exists as the first runnable detector app
- the detector already supports webcam, RTSP, and video-file input
- the detector already supports YOLO inference, overlays, configurable threat classes, and saved evidence under `runs/detect`
- the repo currently includes stock `yolov8n.pt`
- the repo does **not** yet include custom threat weights for `knife` or `gun`
- the repo does **not** yet include a local weapon dataset

Current technical reality:
- the pipeline layer is working or close to working
- true dangerous-object detection for `knife` and `gun` still depends on either custom weights or an external model that already knows those classes
- true behavioral threat detection such as stabbing, armed assault, or theft is not solved by the current detector alone

Execution decision for today:
- prioritize a working dangerous-object path first
- then build a practical first-pass threat logic layer on top of detections and simple spatial or temporal rules
- use local execution first for inference and integration
- use Colab only if custom training becomes necessary and local hardware is too slow or unavailable for same-day delivery

Success criteria for today:
- detect at least one or more dangerous-object classes in a believable demo
- surface an explicit threat state on screen
- save evidence for review
- have a stable fallback demo path using prerecorded video if live webcam behavior is inconsistent

## Checkpoint 2026-04-01 Threat Logic Update
`detector.py` has now been upgraded from a pure class-match alert system into a first-pass threat assessment pipeline.

New current behavior:
- explicit configured classes can still trigger alerts directly
- dangerous-object labels such as `knife`, `gun`, `handgun`, `pistol`, `rifle`, and related aliases can now be grouped through label matching rules
- person labels are now treated separately from weapon labels
- the app now produces higher-level threat states such as:
  - `DANGEROUS OBJECT`
  - `ARMED PERSON`
  - `POSSIBLE ASSAULT`
- threat events now save both raw detections and a structured threat assessment into event metadata

Important limitation at this checkpoint:
- the threat states are currently heuristic and rule-based
- they are not yet learned action-recognition outputs
- the quality of `ARMED PERSON` or `POSSIBLE ASSAULT` still depends heavily on whether the loaded model can correctly detect weapon classes in the first place

Practical implication:
- the software layer is now ready to support a same-day threat demo
- the main remaining bottleneck is obtaining or training a checkpoint that can actually detect `knife` and `gun` reliably enough for the demo footage

## Checkpoint 2026-04-01 Runtime Validation
The local project runtime has now been validated further.

Validated:
- `detector.py` compiles successfully
- project dependencies have been installed locally
- the detector CLI now launches successfully with `--help`
- the local `yolov8n.pt` checkpoint loads successfully through Ultralytics

Critical finding:
- the stock `yolov8n.pt` checkpoint exposes standard COCO-style classes
- that checkpoint is suitable for pipeline proof and person detection
- that checkpoint should **not** be treated as a real `knife` or `gun` detector for this project

Decision update:
- local development and demo integration can continue in this repo today
- Colab is **not** required for the software integration work
- Colab becomes necessary only if the team wants same-day custom weapon weights and does not already have an existing weapon-aware checkpoint available

Same-day delivery framing:
- without custom or external weapon-aware weights, the repo can deliver a strong pipeline demo and a rule-based threat layer
- with custom or external weapon-aware weights, the repo can deliver a much more credible dangerous-object and armed-assault demo

## Checkpoint 2026-04-01 Dual-Model Support
The detector has now been upgraded again to support a more practical same-day demo setup.

New capability:
- the app can now run separate YOLO checkpoints for people and weapons in the same pipeline
- detections from multiple models are merged before threat assessment
- this allows the team to keep strong `person` detection from stock YOLO while also plugging in a custom or external `knife` or `gun` detector

Why this matters:
- many quick same-day weapon checkpoints may only contain `knife` and `gun`
- if the app relied only on that checkpoint, person detection could disappear
- the new dual-model path preserves the `ARMED PERSON` and `POSSIBLE ASSAULT` demo logic even when weapon weights are separate from person weights

Current best architecture for today:
- `yolov8n.pt` or similar for person detection
- a custom or external weapon checkpoint for `knife` and `gun`
- rule-based threat assessment in `detector.py` on top of the merged detections

## Checkpoint 2026-04-01 Weapon Checkpoint Acquired
The project now has a same-day weapon checkpoint available locally.

What was done:
- cloned `zaizou1003/knife_Gun_Detection` into `external/knife_Gun_Detection`
- copied `external/knife_Gun_Detection/exp6/weights/best.pt`
- saved it as `models/weapon_best.pt`
- cloned the official YOLOv5 runtime into `external/yolov5`
- installed the YOLOv5 runtime dependencies locally
- upgraded `detector.py` so it can load legacy YOLOv5 checkpoints through an explicit `--weapon-loader yolov5` path

Validated:
- `models/weapon_best.pt` loads successfully
- the checkpoint exposes classes `{0: 'gun', 1: 'knife'}`
- sample inference from the checkpoint produces weapon detections

Practical consequence:
- the repo no longer depends on Colab just to get a same-day `gun` and `knife` detector
- the detector can now run:
  - stock Ultralytics YOLO for `person`
  - legacy YOLOv5 weapon weights for `gun` and `knife`
  - merged threat logic on top of both

Current recommended demo command:
- `python detector.py --source 0 --weights yolov8n.pt --person-weights yolov8n.pt --weapon-weights "models\weapon_best.pt" --weapon-loader yolov5 --person-classes person --weapon-classes knife,gun --threat-classes knife,gun --show`

## Checkpoint 2026-04-01 Venv Runtime Fix
The first live run in the project `.venv` exposed a missing dependency issue.

Observed runtime error:
- `ModuleNotFoundError: No module named 'pandas'`

Cause:
- the YOLOv5 compatibility path for the legacy weapon checkpoint depends on the YOLOv5 runtime packages
- those packages had been installed in a different Python environment earlier, but not yet in the active project `.venv`

Fix applied:
- installed `requirements.txt` into `.venv`
- installed `external/yolov5/requirements.txt` into `.venv`

Validated in `.venv`:
- `models/weapon_best.pt` now loads successfully
- the checkpoint still reports classes `{0: 'gun', 1: 'knife'}`
- `detector.py --help` also runs successfully from `.venv`

## Checkpoint 2026-04-01 Live Demo Stabilization
The first live camera run exposed two practical demo problems:
- YOLOv5 warning spam flooded the terminal during inference
- one-frame or low-stability weapon detections could still trigger saved threat events

Fixes now added to `detector.py`:
- suppressed the noisy YOLOv5 `autocast` deprecation warnings
- added `--debug-weapon` to print exact weapon detections, confidences, bounding boxes, and source model when they change
- added `--min-threat-frames` so a threat must persist for multiple consecutive frames before it becomes active
- added a `VERIFYING THREAT` intermediate state while the persistence gate is warming up

Practical consequence:
- the terminal is much easier to read during live testing
- false-positive one-frame blips are less likely to create fake threat events
- debugging weapon misfires is now much easier because the exact detections can be inspected in real time

Recommended current testing posture:
- keep `--weapon-conf` relatively strict for early tests
- use `--debug-weapon` while tuning the scene
- keep `--min-threat-frames` above `1` for the live demo

## Checkpoint 2026-04-01 Current Live Status
Current observed live behavior on the webcam:
- the app opens successfully
- person detection is stable
- ordinary objects such as cups are detected by the stock YOLO model
- with stricter settings, no obvious false weapon detections are being shown in the shared test scene
- the terminal is now much cleaner and no longer floods with repeated warning spam

Current interpretation:
- the software pipeline is in a much better place for a same-day demo
- the repo is currently strongest for:
  - `person`
  - `gun`
  - `knife`
  - rule-based states such as `ARMED PERSON` and `POSSIBLE ASSAULT`

Important limitation at this checkpoint:
- the repo does **not** currently have a real action-recognition or violence-recognition model
- it should not yet be described as a reliable detector for:
  - fighting
  - general violence
  - stabbing motion
  - shooting motion
  - assault intent

What it can do right now:
- detect a visible weapon if the weapon checkpoint recognizes it
- detect people
- infer a higher-level threat state when:
  - a weapon is visible
  - the weapon appears attached to a person
  - an armed person is close to another person

What is still needed for stronger behavior detection:
- either a dedicated action-recognition checkpoint
- or a more advanced heuristic layer based on pose, temporal motion, and repeated frame evidence

## Checkpoint 2026-04-01 Violence Heuristics Added
The repo now includes a first-pass pose-based violence layer on top of the existing object and weapon pipeline.

New implementation status:
- `yolov8n-pose.pt` support has been added
- the pose model is now downloaded locally and available in the repo root
- the detector now extracts pose people and keeps short track history across frames
- the detector computes:
  - wrist motion speed
  - arm extension ratio
  - person-to-person proximity
  - weapon-to-hand attachment heuristics

New heuristic threat states:
- `VIOLENCE SUSPECTED`
- `POSSIBLE STABBING`
- `POSSIBLE ARMED ASSAULT`

New runtime controls:
- `--pose-weights`
- `--pose-conf`
- `--violence-distance-ratio`
- `--violence-wrist-speed`
- `--violence-arm-extension-ratio`
- `--weapon-hand-distance-ratio`
- `--violence-min-frames`
- `--debug-violence`

Current technical reality:
- this is still a heuristic violence layer, not a trained action-recognition model
- it is meant to make the same-day demo substantially stronger
- it should be framed as `suspected` or `possible` violence detection, not final semantic certainty

Validated:
- `detector.py` parses successfully
- the new CLI options are available
- the pose model loads successfully
- pose extraction returns people on a test image

## Checkpoint 2026-04-01 Home-Scene Failure Analysis
Home testing exposed a clear pattern:
- the public weapon checkpoint is noisy in the home environment
- false weapon detections appeared on doors, sheets, frame edges, and other background regions
- knife detections were intermittent and often weaker than false gun detections
- the default object detector cluttered the screen with irrelevant labels such as `bird`, `bear`, `cup`, and similar non-threat classes

Fixes now added:
- weapon detections are now validated more strictly before they affect threat logic
- the detector can now reject weapon boxes that are:
  - too small
  - too large
  - hugging the frame border
  - not attached to a person or hand
- the overlay now defaults to relevant detections only instead of drawing every raw object class
- a `--show-all-detections` escape hatch remains available for debugging

New tuning controls:
- `--weapon-min-area-ratio`
- `--weapon-max-area-ratio`
- `--weapon-border-margin-ratio`
- `--allow-unattached-weapons`
- `--show-all-detections`

Current interpretation:
- the software now does a better job of refusing obviously implausible weapon detections
- the weapon checkpoint itself is still the weakest link for accurate knife recognition in real home scenes

## Checkpoint 2026-04-07 GitHub Collaboration Prep
The project has now been isolated for collaboration as its own standalone git repository inside:
- `C:\Users\Demilade\Desktop\CV Threat Intelligence`

Repository prep completed:
- initialized a new local `.git` in the project root so it is no longer tied to the parent Desktop repo for version control
- added a root `.gitignore` to exclude:
  - `.venv`
  - `runs`
  - `__pycache__`
  - editor metadata
  - scratch external repos
- kept the actual runtime assets in scope for collaboration:
  - `detector.py`
  - docs and plans
  - `models/weapon_best.pt`
  - `yolov8n.pt`
  - `yolov8n-pose.pt`
  - vendored `external/yolov5`

Important packaging decision:
- `external/yolov5` is being treated as a vendored runtime dependency for now, because `detector.py` needs it to load the legacy YOLOv5 weapon checkpoint
- its nested `.git` metadata was removed locally so it can be committed as normal project files instead of as an embedded repo/submodule

Current blocker:
- automated GitHub repo creation from the environment failed with a GitHub API owner/auth issue
- local push prep can still be completed, but the empty GitHub repository may need to be created manually from the browser before the first `git push`
