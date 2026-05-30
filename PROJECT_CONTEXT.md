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

Local git status after prep:
- initial commit created successfully
- default branch renamed to `main`
- remote configured as `https://github.com/DEMILADE07/cv-threat-intelligence.git`
- first push attempt failed with `Repository not found`, which confirms the current blocker is repository existence on GitHub, not a local git issue

## Checkpoint 2026-04-08 GitHub Push Completed
GitHub collaboration setup is now live.

Final collaboration state:
- repository exists on GitHub at `https://github.com/DEMILADE07/cv-threat-intelligence`
- local branch `main` is tracking `origin/main`
- local repository is currently clean after push

Implication:
- a collaborator can now be invited directly on GitHub and clone the project without any extra packaging work
- the repo already includes the current POC code, documentation, local model weights, and vendored `external/yolov5` runtime needed for the legacy weapon checkpoint path

## Checkpoint 2026-04-08 Clip Violence Integration
A new same-day clip-based violence layer has now been integrated into `detector.py`.

What was added:
- optional `torchvision` video-model support via `--clip-violence-model r3d_18`
- a rolling frame buffer in the main loop
- periodic clip classification using the pretrained `r3d_18` Kinetics-400 action model
- threat fusion between:
  - person detection
  - validated weapon detection
  - pose heuristics
  - clip-level violence predictions

Current clip-model label mapping:
- `sword fighting` + visible `knife` + multiple people -> `POSSIBLE STABBING`
- `punching person (boxing)` + multiple people -> `PHYSICAL FIGHT`
- `wrestling` + multiple people -> `PHYSICAL FIGHT`
- fight-like clip labels + visible `gun` -> `POSSIBLE ARMED ASSAULT`

Why this path was chosen:
- the environment already has `torch` and `torchvision` installed
- this is much faster than standing up MMAction2 or training a custom video model tonight
- it gives the POC a real motion-based signal instead of relying only on per-frame pose heuristics

Honest limitation:
- this is still generic pretrained Kinetics-400 action recognition, not local fine-tuning
- it is appropriate as a same-day demo accelerator, but not yet the final business-grade violence model

## Checkpoint 2026-04-25 Collaborator Repo Verified
The local repository is now confirmed to be wired to two GitHub remotes:
- `origin` -> `https://github.com/DEMILADE07/cv-threat-intelligence.git`
- `ayo` -> `https://github.com/Ayo-Cyber/cv-threat-intelligence.git`

Repository verification completed:
- `git ls-remote ayo` succeeded, which confirms the collaborator repository exists and is reachable
- `git fetch ayo` succeeded, so the collaborator branch can be inspected directly from this workspace
- local branch remains `main`

Current remote comparison:
- local `main` / `origin/main` are still at commit `4bbd17a`
- collaborator `ayo/main` is at commit `a7b048d`
- the collaborator branch is ahead by 2 commits:
  - `263f5f2` on `2026-04-07`: `feat: add ByteTrack support to detector and include project planning documentation`
  - `a7b048d` on `2026-04-08`: `feat: add ByteTrack tracking, RT-DETR support, and full setup guide`

Meaningful additions observed in the collaborator repo:
- `detector.py` adds ByteTrack-based person tracking through Ultralytics `track(...)`
- `detector.py` adds a `--no-track` flag as a fallback if tracking is unstable or crashes
- `README.md` is expanded with fuller setup guidance for both Windows and Mac
- `README.md` now includes a recommended `RT-DETR` demo path and startup expectations
- new file `48HR_PLAN.md` adds a concrete 48-hour demo execution plan, fallback rules, and demo narrative guidance
- an `ayo_README.md` file also exists in the collaborator branch but appears empty

Practical implication:
- the project is no longer represented fully by the `DEMILADE07` repo alone
- the `Ayo-Cyber` repo should be treated as an active parallel collaboration branch with real implementation changes
- future context, planning, and merge decisions should account for both remotes rather than assuming a single GitHub source of truth

Merge-risk note:
- the collaborator detector changes are narrow but meaningful because they change runtime behavior around tracking
- if these changes are merged later, they should be tested specifically for:
  - webcam stability
  - RT-DETR performance on available hardware
  - ByteTrack failure cases
  - interaction with the existing weapon and pose pipeline

## Checkpoint 2026-04-25 Founder Demo Branch Clarified
The collaborator branch should also be treated as the branch that powered the founder-facing demo path.

Clarified team understanding:
- the version in `Ayo-Cyber/cv-threat-intelligence` is the version reported to have worked well enough for presentation to the founder
- that branch was used as a practical demo-hardening path rather than as a separate product direction

What made that branch useful for the presentation:
- ByteTrack-based person tracking likely made person state handling more stable during the demo
- the README in that branch gave a more operational runbook for setup and troubleshooting
- `48HR_PLAN.md` provided a concrete demo narrative, fallback rules, and clip-based presentation structure
- the branch explicitly documented an `RT-DETR` first option with fallback to `yolov8n.pt`

Important correction:
- the collaborator repo does not include a separate or newer `PROJECT_CONTEXT.md`
- the collaborator branch carries operational demo improvements, not a separate written project history

Current interpretation:
- the project history should recognize that the founder demo relied in part on the collaborator branch
- the main unresolved task is not understanding product direction; it is reconciling the working demo path, current local changes, and the source-of-truth branch strategy

## Checkpoint 2026-05-16 Architecture + GTM Direction Added
Two new product-planning documents now exist in the local workspace:
- `architecture.md`
- `AI_Threat_Detection_GTM_Research.docx`

These documents materially sharpen both the technical north star and the go-to-market wedge.

### Architecture direction now defined
`architecture.md` formalizes a four-part runtime pattern:
- `Agent Mapper`
  - infrequent VLM-based scene understanding
  - outputs `scene_context.json`
- `Detection Core`
  - frame-by-frame CV pipeline
  - outputs `raw_events.json`
- `Customization Engine`
  - applies `user_config.json` rules to raw events
  - produces candidate alerts
- `Verification Gate`
  - VLM confirms or rejects candidate alerts before final escalation

Important architectural implication:
- the long-term product is no longer just a detector with hardcoded alert logic
- it is a context-aware threat platform with:
  - perception
  - raw event generation
  - customer-specific threat policy
  - final AI verification to reduce false positives

### GTM direction now defined
`AI_Threat_Detection_GTM_Research.docx` makes the commercial beachhead much more explicit.

Recommended starting market:
- real estate first
  - gated residential estates
  - commercial real estate
  - malls
  - offices
  - banks / mixed-use as adjacent extensions

Reasoning captured in the GTM document:
- largest addressable market among the options considered
- lowest deployment friction for a software-only product
- existing camera infrastructure already present
- concentrated buyers with recurring security budgets

Important strategic simplification:
- the long-term vision still supports deep customer customization
- but the GTM document recommends that V1 should ship as:
  - one core engine
  - three property presets
  - a fixed 12-rule starter product
- true open-ended custom rule building should be deferred until a later phase

### The 12 rules that the GTM document says must ship in V1
This is now a critical product requirement and should be treated as the V1 rule target set:
1. `Loitering detection`
2. `Perimeter intrusion / fence climbing`
3. `After-hours presence`
4. `Crowd formation`
5. `Running detection`
6. `Tailgating`
7. `Abandoned object`
8. `Unauthorized vehicle / wrong-way movement`
9. `Camera tampering / obstruction`
10. `Person down / fall detection`
11. `Mask / face-covering during business hours`
12. `Power-outage + motion combo`

Important note:
- these 12 rules are now the most important GTM-aligned V1 scope
- the product may eventually support wider rule libraries and full customization
- but V1 success is now strongly tied to shipping these 12 property-security rules

### Mapping against the local codebase
Local `main` remains materially behind this architecture and GTM direction.

What local `main` already aligns with:
- working `Detection Core` conceptually exists
- webcam / RTSP / file ingestion already exists
- object / person / pose / clip-violence pipeline exists
- raw heuristic threat logic exists (`assess_threat`, `assess_violence`)
- evidence saving exists

What local `main` does not yet align with:
- no `ByteTrack` tracking in the active local detector path
- no `TheftDetector`
- no `eval.py` baseline evaluation workflow
- no `Customization Engine`
- no `user_config.json` rule application layer
- no `Verification Gate`
- no `Agent Mapper`
- no scene-context schema or zone model
- no implementation of the GTM 12-rule V1 set beyond a few loosely related primitives

Practical interpretation:
- local `main` is still mostly a POC detector branch with clip-based violence work
- it should not be treated as the best representation of current product direction

### Mapping against the collaborator repo
The collaborator branch `ayo/main` is much closer to the intended architecture than local `main`, but it is still incomplete relative to both docs.

What `ayo/main` already aligns with:
- stronger `Detection Core`
  - ByteTrack support
  - `ViolenceTemporalGate`
  - `TheftDetector`
  - evaluation harness (`eval.py`)
  - ground-truth clips and reports
- explicit product thinking about:
  - RT-DETR as a stronger detector path
  - VLM / Grounding-DINO style verification
  - architecture beyond raw detection
- early theft state-machine work that begins to resemble the event-layer idea in `architecture.md`

What `ayo/main` still does not have:
- no actual `Agent Mapper` implementation
- no `scene_context.json` generation path
- no `Customization Engine` that reads user rule configs
- no `user_config.json` pipeline
- no VLM `Verification Gate` implementation
- no frontend-to-backend threat policy contract
- no explicit zone engine

### Mapping the GTM 12-rule V1 against current implementation
Current reality: the 12 GTM rules are mostly not yet implemented as first-class product rules.

Closest existing building blocks:
- `Loitering detection`
  - not implemented as a formal rule yet
  - could be built from person tracking + dwell time + zone logic
- `After-hours presence`
  - not implemented as a formal rule yet
  - could be built from zone logic + schedules
- `Crowd formation`
  - not implemented
  - could be built from tracked person counts + clustering
- `Person down / fall detection`
  - not implemented
  - pose pipeline could support a first pass
- `Perimeter intrusion / fence climbing`
  - not implemented
  - would need zone + motion logic
- `Tailgating`
  - not implemented
  - would need gate zone + tracking + gate event logic
- `Abandoned object`
  - not implemented
  - would need tracked object persistence + unattended timer
- `Unauthorized vehicle / wrong-way movement`
  - not implemented
  - would need vehicle classes + directional path logic
- `Camera tampering / obstruction`
  - not implemented
  - would need camera-health checks / frame-quality heuristics
- `Mask / face-covering during business hours`
  - not implemented
  - would need face/covering classifier or VLM gate
- `Power-outage + motion combo`
  - not implemented
  - would need integration with lighting/outage state or local brightness heuristics + motion
- `Running detection`
  - not implemented as a standalone rule
  - some motion primitives exist but not as a formal product rule

### New current understanding
The project now has a clearer split between:
- long-term architecture:
  - context-aware, customizable, VLM-assisted threat intelligence
- near-term GTM:
  - real-estate-first
  - fixed 12-rule V1
  - presets before full custom rule building

This resolves an earlier ambiguity:
- the product should be engineered for customization
- but the first sellable version should not be an open-ended infinite rule builder
- it should be a disciplined V1 focused on the GTM 12-rule property-security set

### Recommended execution interpretation from this checkpoint
Near-term priority should now be:
1. treat the collaborator branch as closer to the active product direction than local `main`
2. use the architecture document as the technical north star
3. treat the GTM document's 12 rules as the V1 product contract
4. build missing product layers in this order:
   - schemas / contracts
   - verification gate
   - minimal customization engine
   - zone / dwell / directional event primitives
   - implementation of the GTM 12 rules
5. keep full free-form customer rule authoring as a later phase after the curated V1 rules work reliably

## Checkpoint 2026-05-24 Agent Mapper v1 Landed
Step 1 of the architecture build order is materially complete locally (not yet committed — these files are untracked on local `main`).

### What now exists in the repo
- `schemas/scene_context.schema.json`
  - JSON Schema draft 2020-12, `additionalProperties: false`, 11 required fields
  - bounded enums for `environment_type` (15 values), `suggested_preset` (4), `risk_hints` (15), zone `role` (11)
  - the `risk_hints` enum is a direct 1:1 mirror of the GTM-12 rule universe plus `theft` and `weapon_presence`
- `prompts/agent_mapper_prompt.txt`
  - bounded-vocabulary, JSON-only prompt
  - lists every allowed enum value and the exact output shape
  - explicit "be conservative; structured correctness over creativity" tail
- `docs/AGENT_MAPPER_PLAN.md`
  - frames v1 as scene classifier + preset recommender + risk-hint generator + JSON artifact producer
  - explicitly NOT a per-frame component, NOT an alert decider
  - clarifies that online / local clip testing is a first-class workflow, not just live-camera
- `agent_mapper.py`
  - full v1 implementation, not just a skeleton
  - all 9 modules from the plan present
  - source handling for webcam / RTSP / video file / image file
  - frame sampling: evenly spaced via `CAP_PROP_POS_FRAMES` for seekable video, sequential fallback for live streams
  - representative frame selection via brightness + Laplacian-blur scoring
  - three providers: `mock` (heuristic JSON keyed off camera-id substrings for offline tests), `anthropic` (Claude Vision via raw `urllib`), `openai_compatible` (data-URL image)
  - defensive parsing: `extract_first_json_object` recovers JSON wrapped in prose, hand-rolled normalizers snap invalid enums back to `unknown` / `Unknown`
  - risk hints capped at 5, zones capped at 4, bbox values coerced to non-negative ints
  - outputs land in `runs/context/<camera_id>/{source_frame.jpg, scene_context.json, raw_response.txt}`
- `agent_mapper_smoke.jpg` exists in repo root, suggesting a smoke run has been executed

### Known gaps in the current Agent Mapper v1
- The schema file is loaded but not actually enforced (`_ = schema  # contract parity`) — validation is hand-rolled, so the schema doc and code can drift
- `save_frame=True or args.save_frame` makes the `--save-frame` flag dead code (always saves)
- Hand-validator does not strip unknown keys, so a VLM that returns extra fields will persist them into `scene_context.json` despite `additionalProperties: false` in the schema
- Anthropic provider defaults to `claude-3-5-sonnet-latest`; current Claude 4.x (`claude-sonnet-4-6`) would be both stronger and cheaper for this task
- Nothing consumes `scene_context.json` yet — no Customization Engine, no Verification Gate

### Build-order inversion vs the original plan
The architecture doc recommended: schemas → Verification Gate → Customization Engine → Agent Mapper (last).
The actual local sequence executed: schemas → Agent Mapper (first).

This is defensible — the Agent Mapper is a cleanly isolated component with a stable JSON output contract — but the highest-FPR-leverage piece (Verification Gate) and the GTM-rule-bearing piece (Customization Engine) are still unbuilt.

## Checkpoint 2026-05-24 Customization-As-Source-Of-Truth Clarified
The product framing of the Agent Mapper + Customization Engine pair has been sharpened. This is a meaningful clarification, not just a restatement of `architecture.md`.

### The clarification
- **Agent Mapper is a proposal layer.** It maps the environment, suggests a preset, and emits risk hints. Its output is a *recommendation*, not a verdict.
- **The user's `user_config.json` is the source of truth.** The customer defines their own environment and their own definition of what counts as a threat *in that business context*. The system enforces that definition, not a generic AI judgment.
- **False-positive reduction comes from narrowing.** A generic CCTV AI tries to interpret "is this dangerous?" against the entire universe of possible threats — that is exactly why it over-fires. By contrast, when the user has explicitly said "in my retail shop, the threat surface is: concealment, after-hours presence, abandoned bag at the door," the system only fires on those configured threats. Everything else is by definition not a threat in this deployment.

### Why this matters as a product principle
- it inverts the framing from "the AI decides what's a threat" to "the user decides; the AI helps detect what the user defined"
- it makes the Customization Engine the structural heart of the product, not a polish layer
- it explains why the GTM-12-rule starter set is enough to ship V1 — the user picks from a curated, well-defined surface rather than trusting an open-ended classifier
- it gives the Verification Gate a sharper job: confirm/reject *only against the user's declared rule that fired*, not against a fuzzy general notion of "is this bad?"

### Implication for the next build phase
The Customization Engine is now the right next deliverable, ahead of (or alongside) the Verification Gate, because:
- without it, `scene_context.json` is an orphan artifact with no consumer
- without it, the GTM-12 rules cannot be represented as first-class product objects
- without it, the Verification Gate has no candidate alerts to verify
- the Agent Mapper's `suggested_preset` and `risk_hints` are designed to *seed* a `user_config.json`, but only the Customization Engine turns that seed into running policy

### Concrete near-term deliverables implied by this clarification
- `schemas/user_config.schema.json` — lock the rule contract before code is written against it
- a tiny rule evaluator that reads `user_config.json` and matches `raw_events` against rule triggers
- a clear UX flow: Agent Mapper proposes preset + rule pack → user accepts / edits → result is persisted as `user_config.json` → engine enforces
- the 3 property presets from the GTM doc (Estate Guard, Retail Watch, Office Sentinel) materialised as default `user_config.json` files seeded from the Agent Mapper output

## Checkpoint 2026-05-24 Agent Mapper Stripped To Descriptive-Only
Acted on the principle from the previous checkpoint and Ayo's feedback: the Agent Mapper was emitting threat semantics (`risk_hints`, `suggested_preset`) that belong downstream. Putting threat language in the most upstream layer causes data-quality issues, couples the Mapper to the GTM rule library, and pre-empts the user's authority over their own threat definition.

### What changed
- `schemas/scene_context.schema.json`
  - removed `risk_hints` and `suggested_preset` from `required`
  - removed the `risk_hints` array property (15-value enum gone)
  - removed the `suggested_preset` string property (4-value enum gone)
  - the schema is now purely descriptive: id, source type, environment, description, expected actors, zones, confidence, timestamps, notes
- `prompts/agent_mapper_prompt.txt`
  - removed the "Allowed suggested_preset values" block
  - removed the "Allowed risk_hints values" block
  - removed prompt instructions 4 and 5 (preset choice and risk-hint suggestion)
  - removed `suggested_preset` and `risk_hints` from the example JSON shape
  - added an explicit "Your job is purely descriptive. Do not infer, guess, or list any threats, risks, suspicious behaviors, or threat policy. Threat definitions are handled by other layers and by the user, not by you." block at the top
- `agent_mapper.py`
  - removed `ALLOWED_PRESETS` and `ALLOWED_RISK_HINTS` constants
  - removed `normalize_risk_hints()` function
  - removed preset / risk handling from `parse_and_validate_scene_context()`
  - added defensive `payload.pop("risk_hints", None)` and `payload.pop("suggested_preset", None)` so a non-compliant VLM that still emits these fields gets silently stripped
  - removed risk and preset logic from `mock_scene_context_json()`
  - dropped `suggested_preset` from the CLI summary print
- `docs/AGENT_MAPPER_PLAN.md`
  - rewrote the Purpose section to say what the Mapper deliberately does NOT do
  - removed the Suggested Presets and Risk Hints vocabulary sections
  - removed `risk_hints` and `suggested_preset` from the expected output shape
  - reframed Near-term integration to introduce a separate deterministic Preset Recommender component (no VLM) that maps `environment_type` → preset → default rule pack
  - updated the Summary to say "scene classifier + describer + zone suggester," explicitly NOT a preset recommender or risk-hint generator

### Validated post-strip
- `agent_mapper.py` compiles cleanly under `python -m py_compile`
- end-to-end smoke run against `agent_mapper_smoke.jpg` with `--provider mock` produced a clean stripped `scene_context.json` containing only descriptive fields

### Architectural implication
The right next product layer is now a **deterministic Preset Recommender** (not a VLM call) that consumes `scene_context.json` and a static `presets.json` lookup table and emits a *suggested* `user_config.json` draft for the user to accept or edit. The chain becomes: Agent Mapper (descriptive AI) → Preset Recommender (deterministic lookup) → User (authoritative edits) → Customization Engine (enforcement). Each layer does one job; the threat taxonomy lives in `presets.json`, not in the Mapper.

## Checkpoint 2026-05-24 VLM Testing Phase Set Up
Open-source VLM evaluation harness now exists in `tests/agent_mapper/`. Goal: pick the smallest local VLM that crosses the quality bar for descriptive scene mapping so the project does not depend on a paid API for the Mapper layer.

### Candidate models surfaced by 2026 research
Top open-source multimodal candidates as of 2026-05:
- **Qwen2.5-VL** family (3B / 7B / 32B / 72B) — strong JSON-following, well-tested with Ollama, can produce stable structured output and bounding boxes. Sweet spot for Agent Mapper is 7B.
- **Gemma 3** multimodal (4B / 12B / 27B) — Google's open multimodal line, SigLIP vision encoder, 128K context, strong on document/scene understanding. 12B at Q4_K_M needs ~10 GB VRAM; 27B at Q4_K_M fits a 24 GB GPU.
- **GLM-4.1V-9B-Thinking** and **GLM-4.5V / 4.6V** — Z.ai's newer multimodal line; strong recent leaderboard performance, worth benchmarking if Qwen and Gemma plateau.
- **InternVL 2.5 / 3** — strong vision, sometimes weaker JSON adherence.
- **Pixtral 12B**, **MiniCPM-V 2.6**, **LLaMA 3.2 Vision**, **Phi-4** — fallback candidates for resource-constrained or specialty cases.

Starting set chosen for this project: **Qwen2.5-VL 7B** and **Gemma 3 12B**, run locally via Ollama through the existing `openai_compatible` provider in `agent_mapper.py` (zero code changes — Ollama exposes an OpenAI-compatible endpoint on `http://localhost:11434/v1`).

### Harness layout
```
tests/agent_mapper/
  clips/<env>/<clip_id>.mp4     # user-supplied ground-truth clips (gitignored)
  labels.json                   # one entry per clip with expected env + acceptable alternates
  eval.py                       # multi-model runner; imports agent_mapper.py directly
  results/eval_<run_id>.csv     # per (model, clip) row — gitignored
  README.md                     # one-page setup + runbook
```

### What the harness measures
For each (model, clip) pair, the CSV captures: `valid_json`, `env_match` (exact), `env_acceptable` (in the alternates list), `latency_s`, `leaked_terms_count` and `leaked_terms` (threat-vocabulary tokens that should never appear in a descriptive output — direct check on whether the model is respecting the strip), `scene_description`, `notes`, `error`.

The leak detector uses a hard-coded vocabulary (`threat`, `danger`, `loiter`, `intrud`, `theft`, `weapon`, `assault`, `violence`, `fight`, `tailgat`, `tamper`, `abandoned`, etc.) so violations of the descriptive-only contract surface as a real metric, not just an eyeball pass.

### How to pick a winner
Per the harness README, in priority order:
1. `valid_json` rate — must be ~100% (fundamental contract)
2. `leaked_terms_count` — must be ~0 across the test set (descriptive-only contract)
3. `env_acceptable` rate — should be high; exact `env_match` is the stretch goal
4. `scene_description` quality — manual eyeball pass on a sample
5. Latency and VRAM — only tiebreakers

The Agent Mapper call is infrequent (once per session / every ~5 min per camera), so latency is forgiving. Do **not** default to the biggest model.

### What the user still needs to do
1. Install Ollama for Windows from <https://ollama.com/download/windows>
2. `ollama pull qwen2.5vl:7b` and `ollama pull gemma3:12b`
3. `$env:OLLAMA_API_KEY = "ollama"` (any non-empty string; Ollama doesn't validate)
4. Drop labeled clips into `tests/agent_mapper/clips/<environment_type>/` and add matching entries to `tests/agent_mapper/labels.json`
5. Run `python tests\agent_mapper\eval.py --models qwen2.5vl:7b,gemma3:12b`

End-to-end harness path was smoke-tested with the `mock` provider and runs cleanly (file-not-found on the placeholder clip is the expected behavior until real clips are added).

## Checkpoint 2026-05-24 VLM Candidate Set Corrected
The initial VLM candidate set (Qwen2.5-VL 7B + Gemma 3 12B) was stale. User flagged they had Ollama already installed and access to the much newer Gemma 4 and Qwen 3.5 / 3.6 lines. Re-research with version-specific queries surfaced the actual 2026-current options.

### Updated candidate set (Ollama-supported, vision-capable, local)
- **`qwen3-vl:4b` / `qwen3-vl:8b` / `qwen3-vl:32b`** — dedicated vision-language line, vision properly wired in Ollama. 4B ~2.5 GB Q4 laptop-friendly; 8B is the practical sweet spot at 12-16 GB VRAM; 32B for 24 GB+ machines. 256K native context (extensible to 1M).
- **`gemma4:26b`** — Google's April 2026 multimodal MoE (Gemma 4 26B-A4B, ~3.8B active params over 128 experts), 256K context, vision-native, 140+ languages. Runs at near-4B-dense speed but the weights footprint is ~13-15 GB at Q4.
- **`gemma4:e4b` / `gemma4:e2b`** — edge variants for lower VRAM budgets.

### Models in user's Ollama list that are NOT viable for Mapper testing
- **`qwen3.6`** — vision is **broken in Ollama** right now. The `mmproj` projector ships as a separate file and Ollama's GGUF flow does not wire it up. Text-only would work but image input fails. Would require running via llama.cpp directly with mmproj explicitly loaded, or MLX-VLM on Apple Silicon.
- **`kimi-k2.6:cloud`, `glm-5.1:cloud`, `nemotron-3-super:cloud`, `gemma4:31b-cloud`** — cloud-only Ollama variants. Defeats the "no paid API for Mapper layer" goal of this testing phase.
- **`glm-5.1`** — 754B params, coding/agentic-focused (not vision-first), and the open variant requires far more hardware than a single consumer GPU.

### Revised recommended starting command
```
python tests\agent_mapper\eval.py --models qwen3-vl:8b,gemma4:26b
```

### What changed in the repo
- `tests/agent_mapper/README.md` — model pull commands rewritten with the corrected candidates, explicit warning about Qwen 3.6 vision being broken in Ollama, explicit list of cloud-only Ollama entries that are out of scope, updated example run command.
- No code changes — the harness itself is model-agnostic; only the recommended model names and the README guidance needed updating.

### Lesson logged
Saved as a feedback memory: when researching frontier ML model recommendations, version-specific queries beat "best models 2026" roundups. ML models ship monthly; summary articles lag the actual frontier by 6-12 months. Always assume the user is on the bleeding edge.

## Checkpoint 2026-05-30 Agent Mapper Validated Live On Gemma 4 (OpenRouter)
The Agent Mapper pipeline has now been run end-to-end against a real VLM for the first time, not just the `mock` provider. The decision this session was to wire up OpenRouter + Gemma 4 and prove the path works, deferring real footage until the Mapper itself is confirmed working.

### Hosting / model setup
- Provider path: existing `openai_compatible` provider in `agent_mapper.py`, pointed at OpenRouter (`https://openrouter.ai/api/v1`).
- Model: `google/gemma-4-26b-a4b-it:free` (the validated Gemma 4 26B-A4B MoE, ~3.8B active params).
- API key set as a User env var `OPENROUTER_API_KEY`. Note: a User-scope env var is **not** inherited by an already-running shell, so runs export the key inline in the same process before invoking python.

### OpenRouter account reality
- Account is free-tier with `total_credits: 0`.
- Paid Gemma 4 variant (`google/gemma-4-26b-a4b-it`, no `:free`) returns **HTTP 402 Payment Required** — needs credits.
- The `:free` variant intermittently returns **HTTP 429 Too Many Requests** — this is transient shared-pool saturation, **not** a daily-cap or auth problem (verified: `usage_daily: 0`, auth passes).
- Key finding: **free tier + retries is a viable dev/eval path. Credits are NOT required to proceed.** A ~$5 top-up is purely a "remove the flakiness / unlock paid routing" upgrade.

### Comparable free vision models probed (same size class)
Queried the live OpenRouter models list for free + vision-capable models and tested each against the smoke image:
- `google/gemma-4-26b-a4b-it:free` — **works** (most responsive of the set)
- `google/gemma-4-31b-it:free` — 429
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` — responds but empty content
- `nvidia/nemotron-nano-12b-v2-vl:free` — failed
- `moonshotai/kimi-k2.6:free` — 429

Conclusion: no substitute needed — the originally chosen model is also the best free option currently available.

### Code changes landed in `agent_mapper.py`
- `call_openai_compatible` now has **retry/backoff** (default 3 retries) on 429 and 5xx, respecting `Retry-After`; fails fast on terminal 4xx (401/402). This is what makes the flaky free tier usable.
- Added `read_http_error_body()` helper so HTTP failures surface OpenRouter's actual JSON `error.message` instead of an opaque `HTTP 429`.
- Added OpenRouter routing headers (`http-referer`, `x-title`) — ignored by other backends.
- **Bug fixed:** `camera_id`, `source_type`, `source_frame_path`, and `generated_at` are now **caller-authoritative** — previously the code kept any value the VLM returned, and Gemma 4 hallucinated `generated_at: 2025-01-24T12:00:00Z`. These are facts the caller owns; the timestamp is now system-stamped (verified correct at `2026-05-30T...Z`).
- `agent_mapper.py` compiles clean after all changes.

### What the live run proved (against `agent_mapper_smoke.jpg`)
Even though the smoke image is a placeholder (grey background + black rectangle), the run validated the full contract on a real model:
- valid JSON parse
- correct `environment_type = unknown` (nothing identifiable)
- accurate descriptive `scene_description`
- empty `expected_actors` / `zones`, `confidence = 0.0`
- **zero threat-vocabulary leak** — the descriptive-only contract from the 2026-05-24 strip holds against the live VLM, not just the mock provider

### Still pending (unchanged from prior checkpoints)
- **Real footage** — deferred by user decision this session; only the placeholder image has been run. Real env-classification accuracy numbers require labeled clips in `tests/agent_mapper/clips/<env>/` plus matching `labels.json` entries.
- Nothing downstream consumes `scene_context.json` yet (no Preset Recommender / Customization Engine / Verification Gate).
- The eval harness (`tests/agent_mapper/eval.py`) is ready to run against Gemma 4 via OpenRouter the moment footage lands — same provider/base-url/key/model flags as the validated single-image run.
