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

## Checkpoint 2026-06-10 Retail Shoplifting Named As Lead Use Case + Veesion Research
The founder direction sharpened this session: make the product **scalable for supermarkets / shops to detect shoplifting** — specifically "a customer takes a product from a shelf, conceals it in a bag, the system records the event and triggers an alert." Explicitly positioned against **Veesion.io**. This session was research + codebase audit + approach proposal; no code written yet.

### Veesion / landscape research findings (cited briefing produced this session)
- **Veesion does gesture/action recognition, NOT product-into-bag object tracking.** It classifies the *motion signature* of concealment (reaching to body, stuffing into a bag, tag tampering) on existing CCTV. Confirmed across independent sources (Silicon Republic, White Star Capital), not just their marketing. Detects "10+" concealment gestures; deploys as a small on-prem appliance over RTSP (Hikvision/Dahua/etc.); alerts = short clip to a mobile app for human review; explicitly **no facial recognition / no biometrics**.
- **The "follow the product from shelf into bag" framing is the brittle object-tracking paradigm.** The item is occluded by hand/body/bag exactly when concealment happens (~60% of missed detections are occlusion). Pure item-tracking only works in instrumented stores (smart carts, dense overhead arrays — Amazon Go / Trigo / Caper). **For retrofit on existing CCTV, gesture/action recognition is the approach that actually works.**
- **The only defensible product shape is: record event → surface a short clip → human verifies/decides.** No credible vendor claims autonomous accusation; vision cannot read intent ("concealed it" vs "put it in their own bag while still shopping" can be motion-identical). Human-in-the-loop is mandatory, not optional.
- Field tech stack: video action recognition (SlowFast / X3D / VideoMAE), optionally pose-based (PoseLift / Shopformer — privacy-friendly but only ~60-67% AUC on the one real-store benchmark, i.e. immature). Datasets are thin: UCF-Crime shoplifting subset ~50 clips; DCSASS and PoseLift mostly staged.
- Comparables: **Everseen** (self-checkout non-scan / ticket-switch, vision+POS, ~3s alerts), **Trigo** (verified-loss via MOT + virtual basket vs POS), **Standard AI**, **Caper** (smart carts). Veesion is the closest analog to the stated goal.
- Legal flag (low relevance for Nigeria GTM, noted): French Conseil d'État ruling CE-495153 (2024-06-21) found Veesion's real-time behavioral detection violated GDPR Art. 21. Behavioral surveillance carries regulatory risk in some markets.

### Codebase audit — theft assets already exist on `ayo/main` (NOT on `agent-mapper`)
- **`TheftDetector`** in `detector.py`: per-person **IDLE → APPROACH → ACQUIRE → DEPART** state machine. Uses ByteTrack person tracking + YOLO-Pose wrists + object tracking. Fires `POSSIBLE THEFT` when, after ACQUIRE (wrist inside object bbox for `--theft-acquire-frames`), the object **vanishes while the hand is on it** (strongest signal) or the object moves from its `origin_bbox` / the person leaves the area, then persists in DEPART for `--theft-depart-frames`.
- **Critical limitation:** it tracks only generic COCO classes (`THEFT_OBJECT_CLASSES = backpack, handbag, suitcase, bottle, laptop, cell phone, book, umbrella`) — **not real store merchandise** (small/varied/non-COCO, YOLO won't detect it). No shelf zone, no bag-as-destination concept; "object vanished" is a crude occlusion proxy. This is the brittle object-tracking paradigm. Fine as a **demo/V0 candidate generator**, will not generalize across real supermarkets.
- **`CustomizationEngine`** (`customization.py`) + **`configs/retail_v1.json`** already exist: retail config has a `shoplifting` rule (`trigger: detector=theft, state=DEPART`, priority high) and `loitering_near_merchandise` (`state=ACQUIRE`). Rule-policy layer for retail is skeletoned. `assessments_to_events()` bridges Detection Core → RawEvents → CandidateAlerts. CLI `--mode {all,theft,violence,weapons}` already isolates the theft path.
- **`eval.py`** + `data/test_clips/` already include `theft_shop_01.mp4`, `theft_shop_02.mp4`, `theft_yt_01.mp4` with `data/ground_truth.csv`. Caveat: `week2_baseline.json` theft precision/recall = 1.0 is computed on **one** theft clip — not statistically meaningful.

### Proposed approach (pending founder sign-off — not started)
Do **not** double down on item-into-bag tracking. Layer it the way the 4-layer architecture already implies and the way Veesion effectively works:
1. **Candidate generator (cheap, high-recall):** keep + harden `TheftDetector` as the first-pass trigger (tune for recall). Add a **bag/container + shelf-zone** concept so the trigger is "ACQUIRE → concealment-toward-body," not "COCO object disappeared."
2. **Concealment-gesture signal (the real discriminator):** near-term, reuse the already-validated **VLM (Gemma 4 via OpenRouter / Claude Vision)** to classify the candidate clip zero-shot ("did this person conceal merchandise?") — no training data needed yet. Later (Phase 2): fine-tune a video-action model (X3D / VideoMAE) on Nigeria-relevant staged footage (matches the existing Phase-2 data plan).
3. **Verification Gate = the false-positive killer + human-in-the-loop**, exactly the architecture's existing layer: candidate is the cheap trigger; VLM confirms/rejects against the specific fired rule; only confirmed events escalate.
4. **Output is always a recorded clip for staff review** — never autonomous accusation. `EventRecorder` already saves clips; wire candidate → clip → Verification Gate → alert.
5. **Scalability, both senses:** (a) generalization across stores via Customization Engine presets + VLM verification, not hardcoded thresholds; (b) fleet scale across many RTSP cameras works because the expensive VLM runs only on candidate events (a few/camera/hour), not every frame — the cheap-trigger/expensive-verify asymmetry is the key to multi-camera cost.

### Prerequisite decision flagged to founder
The theft core lives on `ayo/main`; the working branch `agent-mapper` only has the Agent Mapper. Step zero is branch reconciliation. Recommendation: branch a new `theft-retail` off `ayo/main` (richer base) and graft the Agent Mapper work on top, rather than the reverse. Awaiting founder decision on branch strategy + whether to anchor V1 on the VLM-verification path vs investing in a trained action-recognition model.

## Checkpoint 2026-06-10 Decisions Locked + Ayo b992813 Synced + Supervision Adopted
Founder confirmed both decisions: **branch off `ayo/main`** and **anchor V1 on the VLM-verification path** (trained action model deferred to Phase 2). Created local branch **`theft-retail`** off `ayo/main`; restored the 3 planning docs `ayo/main` had deleted (GTM `.docx`, plan `.xlsx`, `docs/OPENROUTER_COSTS.md`). Note: `theft-retail` tracks `ayo/main` — repoint to `origin` before any push.

### Ayo shipped a major commit the same day: `b992813` (2026-06-10) — synced into `theft-retail`
"feat: wire classifier + verification gate into detector, add training pipeline." The **4-layer architecture is now live end-to-end for the first time**: Detection Core → (optional) classifier override → Customization Engine → Verification Gate → record/alert. The Agent Mapper's `scene_context.json` is finally *consumed* (loaded in `detector.py`, passed to both the engine and the gate) — no longer an orphan artifact.
- **`verification_gate.py`** (new, 306 lines): `VerificationGate` with `mock` + `anthropic` (Claude Vision, `claude-sonnet-4-6`) providers; per-rule question templates (`shoplifting`, `violence_in_store`, …); defensive brace-depth JSON extraction + safe fallback; saves per-call artifacts (frame/alert/verdict/raw). Cost-controlled: only fires when the rule signature *changes*, not every frame.
- **Classifier**: `train_classifier.py` fine-tunes **YOLOv8s-cls** on **CamNuvem** (`theft/violence/normal`, 224px) + `download_training_data.py` (yt-dlp) + `infer_video.py`. Reported **FPR 0.67 → 0.034** (mostly a violence win). Wired via `--classifier-weights`.

### Audit findings on `b992813` (what matters for theft)
1. **Everything theft-related is single-frame** — both `run_classifier` (one frame, 224px) and `gate.verify(frame, …)` (one frame to Claude). Concealment is a *motion*; a single frame cannot see it. This is the exact gap the action-recognition layer fills — confirms the plan is aimed right.
2. **The classifier OVERRIDES the state machine** (`detector.py` ~L1811): a single-frame "theft" guess replaces the temporal IDLE→ACQUIRE→DEPART result. Backwards for theft. When the action model lands, **fuse** these signals, don't let one stomp the other.
3. **The gate verifies the frame at the instant the rule first fires** — not necessarily the frame that best shows concealment (grab earlier, conceal later). A **clip/buffer-based gate** (multiple frames; Claude Vision accepts multi-image messages) is a straightforward, high-value upgrade.
4. **Two divergent VLM paths**: `agent_mapper.py` uses OpenRouter/Gemma 4 (`openai_compatible`, *with* retry/backoff); `verification_gate.py` uses Anthropic/Claude (*no* retry). Different keys (`OPENROUTER_API_KEY` vs `ANTHROPIC_API_KEY`). Unify later; not urgent.

### Supervision adopted (the X-post lead — roboflow/supervision, 40k stars)
Decided to use **Supervision 0.28.0** for spatial primitives (zones, annotation), taking *tracking from Ultralytics* (`sv.ByteTrack` is deprecated in 0.28, removed in 0.30). Added `supervision>=0.28.0` to `requirements.txt`. New scaffolding (untracked, this branch):
- **`retail_zones.py`** — `RetailZoneMonitor`: Ultralytics detect+track → `sv.Detections.from_ultralytics` → named `sv.PolygonZone` membership + per-track **dwell** accounting (enter/leave timestamps, optional per-zone loiter threshold) + annotation. Class-agnostic; CLI demo filters to persons. This is the **spatial foundation the action model plugs into** ("WHO is WHERE, for HOW LONG" — it does NOT decide theft).
- **`configs/retail_zones.example.json`** — example shelf/exit polygons; documents that zones are camera-specific pixel coords.
- **`tests/test_retail_zones.py`** — 6 checks (presence, dwell accumulate+alert, reset-on-leave, untracked handling, config load, annotate). **All passing** on a torch-free venv (synthetic `sv.Detections`), so the Supervision API usage is verified without needing YOLO installed.

### Validated on real footage (2026-06-10)
Installed `ultralytics` 8.4.64 + torch 2.8.0 (MPS available) into the Mac `.venv` and ran `retail_zones.py` end-to-end on `data/test_clips/theft_shop_01.mp4` (360×640 portrait CCTV, 931 frames, 30fps — a woman lingering at a wig/hair-product shelf for the full 31s). The full pipeline works: detect → ByteTrack (auto-installed `lap`) → `sv.Detections.from_ultralytics` → `PolygonZone` membership → dwell → annotation. Annotated frames render zones, tracked boxes, dwell labels, and in-zone counts correctly.

**Two real-world findings (tuning items, not blockers):**
1. **Mannequin heads / wig displays get detected as `person`** by `yolov8n` → spurious in-zone detections (`shelf_right` presence_frames=1001 > 931 total frames means ≥2 "people" in zone on many frames). Mitigations: higher conf, person size/aspect filtering, a stronger model (yolov8s/m), or restrict the zone to the floor-standing area.
2. **ByteTrack ID fragmentation**: ~1 real shopper produced **19 unique track ids** (occlusion by the caption banner + reflections + mannequins). Because dwell resets on ID change, the 8s loiter alert never fired despite 31s of real presence (max unbroken dwell only 6.1s). Mitigations: raise `track_buffer` in `bytetrack.yaml`, a stronger detector, or make dwell "sticky" (bridge brief track-loss gaps / re-associate). **This matters for the action model too** — fragmented tracks break any per-person temporal window, so track stability is a prerequisite, not a nicety.

(The demo zone config used was clip-specific, `/tmp/zones_theft01.json`, CENTER anchor because the caption banner occludes feet; the committed `configs/retail_zones.example.json` stays the generic template.)

## Checkpoint 2026-06-10 Track-Stability Pass (gates the action model)
Did a measured tuning pass to fix the ID-fragmentation found above, because stable per-person tracks are a prerequisite for any temporal action model. Three levers, all landed in `retail_zones.py` + configs:
1. **Retail-tuned tracker** `configs/bytetrack_retail.yaml` — `track_buffer` 30→120 (~4s; a shopper occluded briefly reclaims her original id), `new_track_thresh` 0.25→0.50 (weak/mannequin detections don't spawn ids), `track_high_thresh` 0.30.
2. **Person-plausibility filter** `filter_person_detections()` — drops boxes that are too small or not taller-than-wide (mannequin heads, wig displays, reflections). Defaults: min area 1.2% of frame, min aspect h/w 1.10.
3. **Sticky dwell** — `RetailZoneMonitor(dwell_grace_seconds=…)` bridges brief zone/track dropouts (boundary jitter, 1-frame loss) so dwell doesn't reset on a flicker. Default grace 1.5s in the demo.

### Measured before/after on `theft_shop_01.mp4` (931 frames, 31s, one real shoplifter)
| metric | BEFORE (yolov8n, default bytetrack, conf 0.3) | AFTER (tuned bytetrack + filter + grace 1.5s, conf 0.4) |
|---|---|---|
| unique track ids | 19 | **9** |
| shelf_right presence_frames | 1001 (>931 = mannequins counted) | 679 (<931, plausible) |
| shelf_right max_dwell | 6.1s | **9.3s** |
| loiter alert (8s threshold) | never fired (0) | **fired (40 frames)** |

The loiter signal now actually fires on the real clip. Mannequin false positives are gone (person filter). IDs roughly halved.

### Tracker comparison logged
- Tuned **ByteTrack** (`bytetrack_retail.yaml`): 9 ids, max_dwell 9.3s, loiter fired — **chosen default** (lighter for edge).
- Default BoT-SORT: *worse* (12 ids, 7.9s, loiter didn't fire) — its defaults aren't tuned and ReID is off.
- Tuned **BoT-SORT + ReID** (`configs/botsort_retail.yaml`, GMC off for fixed CCTV): 9 ids, max_dwell 10.0s, loiter 59 frames — marginally better dwell continuity, more compute. Kept as the higher-accuracy option.
- **Remaining ~9 ids is detector-bound, not tracker-bound.** Pushing toward single-id needs a stronger detector (`yolov8s/m`); deferred — current stability is enough for the loiter signal and to proceed.

### Tests
`tests/test_retail_zones.py` now 9/9 passing (added: sticky-dwell-bridges-gap, dwell-resets-after-grace, person-filter-drops-mannequin). All torch-free (synthetic detections).

## Checkpoint 2026-06-10 Action Layer v1 — Pose-Based Concealment Detector
Built the action-recognition layer (`concealment.py`), the piece that makes the theft signal *temporal* (Veesion-style gesture recognition) instead of single-frame. Per the locked V1 decision (VLM anchor, training deferred to Phase 2), this is a **transparent heuristic over a per-person skeleton sequence — NOT a trained model** — and a recall-oriented **candidate generator** for the Verification Gate, NOT a final verdict.

### How it works
- Per-track rolling window (1.2s) of skeletons (uses the now-stable tracks from the track-stability pass + `yolov8n-pose.pt`; adds **hip** keypoints, which detector.py's violence pose code lacks — essential for the hand-to-waist signal).
- Three normalised temporal features (scale-invariant via shoulder↔hip body scale):
  - `f_waist` — nearest hand got close to the hip/waist line
  - `f_retract` — a hand reached OUT laterally then pulled IN to the torso and ended low (the conceal motion). Uses lateral offset from the torso centerline (a corrected feature — raw shoulder-to-wrist distance does NOT separate "reach out" from "hand at waist", since reaching to your own hip is also a long span).
  - `f_dwell` — a hand lingered at the waist (stuffing into pocket/waistband)
- Weighted score (0.40/0.30/0.30) → persistence gate (`min_candidate_frames`) → `candidate` bool with human-readable reasons.
- **The seam:** `score_window()` is the swappable head — replace its body with a trained pose-sequence classifier (LSTM/1D-CNN/transformer) on the same feature vectors in Phase 2, nothing else changes.
- Graceful occlusion handling: if hips are never visible, sets a `limited` flag and degrades the score rather than firing blind.

### Validated
- `tests/test_concealment.py` — 5/5 passing, torch-free synthetic skeletons: concealment-motion fires (score 0.91), normal browsing stays at 0.00, occluded-hips degrades to 0.09+`limited`, empty window = 0, per-track state cleanup.
- **Real clip `theft_shop_01.mp4`:** the detector **correctly fired `CONCEAL-CANDIDATE` on the actual shoplifting motion** — track #1 ramped 0.30→0.60→sustained 0.6–0.85 (peak ~0.85), with `f_waist` up to 0.84, `f_retract` up to 0.81, `f_dwell` 1.0, and interpretable reasons. (YOLO-pose estimated hip positions even partly behind the caption banner, so the waist signal survived.)

### Honest caveats (do not overstate)
- This is **one positive clip**. The detector firing here is encouraging, NOT validation. **No normal/negative clips have been run through it**, so precision / false-positive rate is unmeasured. A "hands at waist for a while" heuristic will also fire on adjusting clothes, phone-at-waist, hands-in-pockets — which is exactly why the **VLM Verification Gate must filter these**; precision is the gate's job, recall is this layer's job.
- Depends on `yolov8n-pose` hip estimates, which are noisy under occlusion. A stronger pose model would help.

### Files added this session (untracked on `theft-retail`, nothing committed yet)
`retail_zones.py`, `concealment.py`, `configs/retail_zones.example.json`, `configs/bytetrack_retail.yaml`, `configs/botsort_retail.yaml`, `tests/test_retail_zones.py`, `tests/test_concealment.py`; `requirements.txt` gains `supervision>=0.28.0`; restored planning docs; `PROJECT_CONTEXT.md` updated. A `.venv` exists locally (gitignored) with ultralytics+torch+supervision.

### Next phase (integration — needs go-ahead)
Wire the three V1 pieces into one trigger in `detector.py`: **shelf-zone interaction (`retail_zones`) + concealment candidate (`concealment`) + the existing TheftDetector state machine → fused candidate → rolling clip buffer → multi-frame Verification Gate (upgrade `verification_gate.verify` from 1 frame to a short clip) → alert**. Then measure precision on normal-shopper clips (need negative footage) and recall on more theft clips. Also revisit: fuse (don't let the single-frame classifier override the temporal signal); unify the two VLM code paths.

## Checkpoint 2026-06-11 Zone→Customization Wiring + Bag/Trolley Robustness
Founder gave two build directives and an important conceptual question. Conceptual clarification logged: the standalone scripts (`concealment.py`, `retail_zones.py`) each exercise ONE module in isolation — running `concealment.py` only shows concealment because that's all it loads. The full system is `detector.py` (the "orchestra") which runs weapons + violence + theft + (soon) concealment + zones every frame and feeds the Customization Engine → Verification Gate.

### Part A — Zone → Customization wiring (unlocks the bank example + GTM property rules)
The architecture always envisioned zone+time+dwell rules (architecture.md even has `loitering_at_atm` / `after_hours_intrusion`), but zone data never flowed into the engine. Closed that gap:
- **`customization.py` → `zone_states_to_events()`**: bridges `RetailZoneMonitor` output into `presence` RawEvents carrying `zone`, `dwell_seconds`, `loitering` in `extra` (duck-typed, no CV import). Now a rule's `context_filter` can read `zone == 'vault'` / `zone == 'aisle' and dwell_seconds >= 8`, and `time_filter` scopes it to after-hours.
- **Configs**: `configs/bank_zones.example.json` (geometry: vault + atm) and `configs/banking_zones_v1.json` (rules) implement the founder's exact example — `vault_after_hours` = presence in vault zone + `time_filter 20:00-06:00` → CRITICAL. Plus `configs/retail_zones_rules.example.json` for the retail demo.
- **`retail_zones.py` demo** gained `--rules` and `--simulate-time HH:MM`: loads the engine, converts zone presence to events per frame, prints `[RULE FIRED]`. **Verified live** on `theft_shop_01.mp4` at simulated 21:00 → `after_hours_shelf (HIGH)` and `shelf_loitering (MEDIUM)` fired.
- **`tests/test_zone_customization.py`** (4/4): vault rule fires at 9pm and is silent at noon (time filter), atm loitering needs 75s not 20s (dwell), unconfigured zone fires nothing.
- **Customization answer to founder:** YES, the system is genuinely customizable and the bank example works now at the logic level. Same engine serves every vertical — bank uses zone+time rules, supermarket uses concealment+zone rules, all from `user_config.json`. Remaining for a *non-developer*: a frontend to "draw the box" (today the box is JSON, which is exactly the contract that UI would emit). This same wiring unlocks most GTM-12 property rules (loitering, after-hours, intrusion) at once.

### Part B — Robust concealment: destination classification (bag vs pocket vs trolley)
Founder's insight: concealment is a *destination* problem — pocket/bag = threat, trolley = normal (they pay at the counter). v1 was waist/pocket-biased. Extended `concealment.py`:
- Added **`f_bag`** feature + `hand_to_bag`/`hand_at_bag` via point-to-bbox distance to detected **personal-bag** boxes (COCO `backpack`/`handbag`/`suitcase`, ids 24/26/28). Concealment destination = `max(f_waist, f_bag)`; assessment now carries `destination ∈ {waist, bag, None}`.
- **Trolley is safe by construction**: a shopping cart/basket is NOT a COCO bag class, so it never produces a bag bbox and never fires `f_bag` — putting goods in a trolley yields no destination. Documented explicitly.
- `update(pose_frames, ts, bag_bboxes=...)` (backward-compatible default None); demo loads a second YOLO object model (`--object-weights`, `--no-bags`) and feeds bag boxes; overlay shows `CONCEAL>BAG` / `CONCEAL>WAIST`.
- **Honest limit**: the hard pocket-vs-bag-vs-trolley *edge cases* (reusable shopping bag they'll pay for; basket misdetected as handbag) are still the **VLM gate's** job — pose = recall, gate = precision. And "are personal bags a threat here?" is itself a per-business `user_config` setting (boutique vs cash-and-carry).
- **`tests/test_concealment.py`** now 7/7: bag concealment fires (`dest=bag`, 0.88), trolley stays safe (`dest=None`, 0.05), pocket still fires (`dest=waist`).

### Status: 20/20 unit tests passing (concealment 7, retail_zones 9, zone_customization 4). All modules compile; demos run on real video. New/changed this session: `customization.py` (+zone bridge), `concealment.py` (+bag/destination, +viz overlay, +bag demo), `retail_zones.py` (+--rules/--simulate-time), configs `bank_zones.example.json` / `banking_zones_v1.json` / `retail_zones_rules.example.json` / `retail_zones.theft_shop_01.json`, `tests/test_zone_customization.py`. Still nothing committed (all on `theft-retail`, untracked/modified).

### Next (needs go-ahead): full `detector.py` integration
Wire zones + concealment (with destination) + state machine into one fused trigger → rolling clip → multi-frame Verification Gate → alert, driven by `user_config.json`. Then measure precision on normal-shopper footage (still needed).

## Checkpoint 2026-06-11 Full Pipeline Integrated (the orchestra) — pushed earlier work first
Pushed the prior session's work to `origin/theft-retail` (Demilade's repo) as commit `7c56ed7` (brief 2-line message, no co-author line, per founder) — a new branch carrying Ayo's full work + ours; `main` untouched. Then built the full integration.

### Architecture decision: a dedicated `retail_pipeline.py`, not an edit to `detector.py`
Chose to compose the V1 pieces into a NEW orchestrator rather than surgically editing Ayo's 1827-line `detector.py` (he's actively working on it; editing risks merge conflicts + the sv.Detections-vs-his-Detection-dataclass friction). `retail_pipeline.py` REUSES the shared layers (`CustomizationEngine`, `VerificationGate`) so there's no duplication of the contract logic — only the retail-specific loop is new. Can be folded into `detector.py` later, or `detector.py` refactored to call the shared components.

### What the orchestra does (all wired, runs on real video)
`retail_pipeline.py`: one YOLO-pose model gives person boxes + track ids + keypoints →
- `RetailZoneMonitor` (shelf zones + dwell) → `zone_states_to_events` → `presence` events
- `ConcealmentDetector` (pose action + bag/pocket destination; optional object model for bags) → `concealment_to_events` → `concealment` events
- merged → `CustomizationEngine.evaluate(user_config rules)` → candidate alerts
- top alert → `VerificationGate.verify` (mock | anthropic Claude Vision), **throttled to a new rule match** so the VLM runs per-event not per-frame
- on confirm → rolling-buffer **evidence clip** (`runs/retail/event_XXXX/clip.mp4`) + `alert.json` (rule + detector + person + verdict)
- live overlay: shelf zones, per-person concealment score / `CONCEAL>WAIST|BAG`, red ALERT banner.
- New converter `concealment_to_events()` in `customization.py`; rules in `configs/retail_pipeline_v1.json` (`shoplifting` on concealment, `shelf_loitering` on dwell). Rule named `shoplifting` so the gate asks the right question.

### Verified end-to-end on `theft_shop_01.mp4` (mock gate)
931 frames → **3 confirmed `shoplifting` alerts** (concealment, destination=waist), each with a saved evidence clip + `alert.json` + gate artifacts (frame/verdict/raw). Annotated mp4 produced. The mock gate confirms everything, so the 3 alerts include the real concealment + track-fragmentation/mannequin false positives — which is exactly what a REAL VLM gate is there to filter. **21/21 unit tests pass** (concealment 7, retail_zones 9, zone_customization 5 incl. a new concealment→shoplifting-rule wiring test).

### Honest status / what's left
- **Precision is still unmeasured.** With mock gate everything confirms; we need (a) a real Anthropic key to run `--gate-provider anthropic`, and (b) **normal-shopper (negative) footage** to measure false-positive rate. This is the single most important open item before any "it works" claim.
- The Verification Gate is still **single-frame** (best frame at alert time); multi-frame clip verification is the next quality upgrade.
- `retail_pipeline.py` is uncommitted (new, on `theft-retail`). The theft state-machine (`detector.py`) is NOT folded in — concealment + zones are the retail signal; the state machine can be added as an extra signal later.
- New/changed: `retail_pipeline.py` (new), `customization.py` (+concealment_to_events), `configs/retail_pipeline_v1.json` (new), `tests/test_zone_customization.py` (+1 test).

## Checkpoint 2026-06-12 Real VLM Gate Wired (OpenRouter) + Multi-Frame Fix + Model-Quality Finding
Ran the real Verification Gate against a live VLM for the first time, and learned what actually drives gate quality.

### OpenRouter gate provider added
- `verification_gate.py` now supports an `openrouter` provider (alongside `mock`/`anthropic`), reusing `agent_mapper.call_openai_compatible` (proven retry/backoff). Defaults: model `google/gemma-4-26b-a4b-it:free`, key env `OPENROUTER_API_KEY` (auto-selected when provider=openrouter). `retail_pipeline.py` gained `--gate-provider openrouter`, `--gate-model`, `--scene-description`, `--environment-type`, `--gate-frames`.
- **Robustness fix:** a transient gate/API error no longer crashes the pipeline — it logs `[gate error] … (alert held, not raised)` and continues. (Found because a 429 killed the whole run.)

### Free-tier reality (validated live)
- Key valid; the earlier 401 was PowerShell `$env:` syntax used in zsh (should be `export`). 
- **Gemma 4 `:free` is heavily 429-rate-limited** right now (shared pool). Among free vision models, **`nvidia/nemotron-nano-12b-v2-vl:free` responds** and was used for testing.

### Multi-frame gate built — and the real lesson
- Implemented best-frame selection: the pipeline keeps a per-frame `{track_id: concealment_score}` buffer and sends the gate the **clearest frames** (highest score), not the first (often blurry) one. `verify()` + both providers + `agent_mapper.call_openai_compatible` now accept a list of frames; the Anthropic/OpenAI payloads carry multiple images; the prompt says "frames from the same short event."
- **Empirical finding (important):** with the small free model, **3 frames made it judge "standing still across stills" and reject**; **1 single best frame + scene context CONFIRMED** the real concealment (person #5, **0.85**, "bending over, hands near waist in a merchandise area"). So for a weak model, `--gate-frames 1` + a primed `--scene-description` is the working config; a stronger model would likely benefit from more frames.
- **Precision is excellent, recall is model-bound.** The gate reliably rejects normal behavior (great FP control); confirming subtle concealment on grainy 360p CCTV needs a stronger model. This matches the day-one research (subtle concealment is hard; small models + heuristics aren't enough for production recall). Levers: (1) Gemma 4 with ~$5 OpenRouter credit (no 429) or Claude via `--gate-provider anthropic`; (2) the Phase-2 trained action model + clearer/staged footage.

### Status
End-to-end with a real VLM now yields a confirmed shoplifting alert (0.85) + saved evidence, while rejecting normals. 21/21 unit tests still pass. Track fragmentation still splits the shoplifter across ids #1/#5/#7 (id #5 caught the confirmable moment). Uncommitted on `theft-retail`: `verification_gate.py`, `agent_mapper.py`, `retail_pipeline.py` changes. Security: a paste of the OpenRouter key occurred in chat — **rotate it**; the on-disk copy was removed. `detector.py` unification still queued.

## Checkpoint 2026-06-12 Unification — zones + concealment folded into detector.py (ONE system)
Done on a dedicated branch **`unify-detector`** (off `theft-retail`, after the gate commit `f88c840` was pushed) — isolating the edit to Ayo's 1827-line `detector.py` so `theft-retail` stays stable and the change is reviewable.

### What changed in `detector.py` (all additive / opt-in)
- **Hip keypoints**: added `left_hip`/`right_hip` to `POSE_KEYPOINT_INDEX`, to `PosePersonState` (defaulted, so the violence path is unaffected), and to `extract_pose_people`. The concealment hand-to-waist signal needs these; nothing else does.
- **Two adapters** so the retail layer rides the existing single pose pass (which already carries a track id from `assign_pose_tracks` — no second tracker): `pose_people_to_sv_detections()` (→ `sv.Detections` for the zone monitor) and `pose_people_to_concealment_frames()` (→ `concealment.PoseFrame`). Bag boxes for the concealment destination come from the existing object `detections` (filtered to `CONCEALMENT_BAG_CLASSES`) — no extra model.
- **Flags**: `--zones <geometry.json>`, `--concealment`, plus gate upgrades `--gate-provider openrouter` and `--gate-model`. All retail features are **off by default**; lazy imports mean `supervision`/`retail_zones`/`concealment` load only when used.
- **Wiring**: in the existing `if args.config:` block, zone presence + concealment events are appended to the same `raw_events` that `assessments_to_events` produces, so the Customization Engine + Verification Gate now see **weapons + violence + theft + concealment + zone** signals in one stream.

### Verified
- Unified run on `theft_shop_01.mp4` (`--zones --concealment --config configs/retail_pipeline_v1.json --gate-provider mock`) fired BOTH new event types: `shelf_loitering (PERSON IN ZONE SHELF_RIGHT)` and `shoplifting (POSSIBLE CONCEALMENT (waist))`, confirmed by the gate — alongside the existing detectors. **This is the single orchestra.**
- Baseline run **without** `--zones/--concealment` on a violence clip: loads and runs clean, no retail events, no errors — Ayo's path preserved.
- `detector.py` compiles; new flags show in `--help`; 21/21 unit tests still pass.

### Status / next
- One command now runs the whole product, routed per customer by `user_config.json` (retail uses concealment+zone rules; bank uses zone+time rules; weapons/violence always available). Same backend, many verticals.
- Uncommitted on `unify-detector`: `detector.py` + this `PROJECT_CONTEXT.md`. To merge into `theft-retail`/`main` later (coordinate with Ayo since it touches his file).
- Still open (unchanged): recall needs a stronger gate model (Gemma 4 paid / Claude) + Phase-2 trained action model; track fragmentation; multi-frame gate not yet wired into detector.py's gate call (single-frame there for now — `retail_pipeline.py` has the multi-frame/best-frame version).

## Checkpoint 2026-06-22 Real Gate Validated + Anomaly Batch Test (technical deltas since unification)
Tested the unified system with the real VLM gate (OpenRouter) and across a real robbery dataset.
- **Real OpenRouter gate works.** Added an `openrouter` provider to `verification_gate.py` (reuses `agent_mapper.call_openai_compatible`; defaults to `google/gemma-4-26b-a4b-it:free`, env `OPENROUTER_API_KEY`). Gemma's free tier is 429-saturated; `nvidia/nemotron-nano-12b-v2-vl:free` responds. The gate **reasons correctly** — it rejects blurry/ambiguous candidates ("the image is blurry…", "no clear evidence of violence") = good precision, conservative recall. Multi-frame/best-frame + scene-context priming added in `retail_pipeline.py`.
- **Robustness fixes:** `detector.py`'s gate call is now wrapped in try/except (`[gate error] … alert held`), and `agent_mapper.call_openai_compatible` retries the free tier's empty "no choices" responses. (Found via live crashes.)
- **CamNuvem robbery batch test** (49 "anomaly" clips a friend shared — the same CamNuvem robbery dataset Ayo already uses; downloaded to `data/anomaly/`, gitignored). New reusable harness `tools/batch_anomaly.py` (single model load, reuses detector.py wiring + adds concealment) + `configs/all_threats_v1.json`. Result across 49 clips (stride 2, mock gate): **ANY THREAT 75%** (37/49) — concealment 75%, violence 51%, weapon 28%, theft 10%. All clips are robbery-positive, so this is **recall only** (no FP rate — needs normal clips + real gate). Confirms the system runs across varied real low-res CCTV and the concealment signal is very active.
- Uncommitted on `unify-detector`: `detector.py` (gate try/except), `agent_mapper.py` (empty-retry + multi-image), `tools/batch_anomaly.py`, `configs/all_threats_v1.json`.

## Checkpoint 2026-06-22 DEPLOYMENT DIRECTION LOCKED — Edge-First + On-Device VLM (read this for the product/infra picture)
Founder + co-collaborators set the deployment architecture. This reframes everything downstream of the detector.

### Team & roles
- **Demilade** (this workspace) — ML core: detection + concealment + zones + the gate.
- **Ayomide "Ayo" Atunrase** — executables / Docker / bundling / cameras. Already bundled a **Mac app `CVTI-0.1.0-mac.dmg` (~316 MB, in repo root, gitignored)**; Windows `.exe` in progress.
- **Martins ("Martin's Nc")** — WhatsApp bot + API + customization integration + hardware procurement. (Demi Ezylag also in the group.)

### Architecture (decided, firm)
- **EDGE-FIRST, NO CLOUD.** Inference runs **on-site on the device wired to the cameras**. **Only alerts leave the box — never video.** A server exists only for **software updates** and the **WhatsApp link** to customers. ("We are using edge, we can't use server.")
- **Sell device + software together.** Production hardware = **NVIDIA Jetson Orin Nano Super Dev Kit** ($249 / ~₦500k; **8 GB shared CPU/GPU RAM**). **PC for demos.**
- **Cameras:** WiFi/RTSP (ONVIF), 2–3 live streams. (`detector.py` already supports RTSP.)
- **Interface = WhatsApp bot, NO mobile app for the MVP** — customers **customize the system AND receive alerts** via WhatsApp. **`user_config.json` is the contract** between Martins' WhatsApp bot and the executable — Martins needs that structure from us (we already have it: the Customization Engine schema).

### HARD CONSTRAINTS
- **Total package ≤ 4–5 GB**, and it must **fit the Jetson's 8 GB RAM at runtime** — detector + VLM + everything. (Both disk size AND memory footprint.)
- **The VLM must run LOCALLY, fully offline, while staying accurate.** This is the central technical challenge and the team's current focus.
- **YOLO must be exported to TensorRT** for production (no raw PyTorch `.pt` inference on the Jetson).
- **Dockerize everything** → one-command deploy to a new site. Ship as an **executable**.

### Governance / DevOps baseline (this week, per Ayo's DevOps contact)
- Pinned `requirements`, **protected `main`**, **GitHub Actions CI running tests on every PR** before merge.
- **Model registry**: every training run logs dataset/metrics/**FPR**/weights; **no model ships to a customer without a validated FPR attached** (hard rule).
- **Balena.io-style fleet management** later (can't hand-update >3–4 sites).
- **Feedback loop = the differentiator**: every false positive + missed detection logged and **auto-queued for retraining**. Core feature from day one. (Our gate already saves per-event artifacts — the foundation for this.)

### The on-device VLM pivot (critical for our part)
- **Our gate currently calls OpenRouter (cloud) → conflicts with offline edge.** It must re-point to a **LOCAL small VLM**. Good news: the gate's `openai_compatible` provider works **unchanged** against a local **Ollama** server (OpenAI-compatible at `localhost:11434/v1`) — only the base URL + model name change. The OpenRouter work transfers directly.
- The cloud **Gemma-4-26B-A4B is too big** to quantize/run on a Jetson (~13–15 GB at 4-bit). Need **2–4B** models, quantized (~1.5–2.5 GB at 4-bit). **On-device shortlist being evaluated:** `Gemma-4-E4B-it` (edge variant; strong reasoning+factuality), `Qwen-3.5-VL-2B`, `SmolVLM2-2.2B` (best video/temporal), `VILA1.5-3B` (likely weakest, older).
- **Current task (Demilade):** test these candidates **on OUR task** (gate verification on our clips) to pick a winner — NOT on generic MMMU/MMLU benchmarks. 4-bit quantization's accuracy hit is small (~1–3%); model choice + task-fit matters more; must verify the winner's **vision works in Ollama on the Jetson** (some VLMs' vision projector isn't GPU-accelerated there).

### How this maps to what we've built
- Most of our pipeline (YOLO, pose, concealment, zones, Customization Engine, `user_config.json`) is already **local/edge-friendly**. The **gate is the one cloud piece to localize.**
- Near-term ML-core deliverables implied: (1) **local-VLM gate** via Ollama + pick the on-device model; (2) hand Martins the **`user_config.json` structure**; (3) **TensorRT export** of YOLO; (4) wire the **feedback-loop logging** (FP/miss → retrain queue) onto the existing gate artifacts.

## Checkpoint 2026-06-24 Local-VLM Gate Built + On-Device Model Bake-Off (HANDOFF TO AYO — read this)
Executed the local-VLM gate pivot from the deployment direction, and built a harness to pick the on-device gate model. **All this is on branch `unify-detector` and now pushed.**

### What's built
- **Local Ollama gate** (`verification_gate.py`): added an `ollama` provider (offline, OpenAI-compatible `localhost:11434/v1`) — this is the **production edge-gate path**. Just `--gate-provider ollama --gate-model <tag>`. The OpenRouter `openai_compatible` work transferred directly; only the base URL/model change. Also added a **`--cot`** (chain-of-thought) prompt path and multi-frame gate (`verify()` accepts a list of frames; Anthropic/OpenAI payloads carry multiple images).
- **Gate model bake-off** (`tools/gate_bakeoff.py` + `tools/gate_bakeoff_labels.json`, doc `docs/GATE_MODEL_BAKEOFF.md`): rule-aware (concealment / violence / weapon) — picks the peak frame for each rule using the real detectors, asks the gate the matching question, scores recall/specificity/accuracy/JSON/latency + a per-rule breakdown. **Three A/B toggles** to improve the gate: `--gate-frames N` (multi-frame spanning the motion), `--cot`, `--use-agent-mapper` (grounds the gate with a real Agent-Mapper scene description, run locally via Gemma). `--models mock` tests wiring without Ollama.
- **Robustness** (found via live crashes): `detector.py`'s gate call wrapped in try/except (`[gate error] … alert held`); `agent_mapper.call_openai_compatible` retries the free tier's empty "no choices" responses.
- **Reusable batch harness** `tools/batch_anomaly.py` (single model load) + `configs/all_threats_v1.json` (full-system rules).

### Key findings (on a SMALL, partly-noisy set — directional)
- **`gemma3:4b` is the best gate so far** — on the clean *concealment* slice it scores **100% recall + 100% specificity**, grounds well, and is **right-sized for edge (~3.3 GB)**.
- **`qwen2.5vl:3b` is unsuitable** — rubber-stamps "yes" (0% specificity), even **hallucinated a person in an empty warehouse**.
- **`gemma4:e4b` is 9.6 GB → too big for the 4–5 GB edge budget.** So the team's shortlist Gemma won't fit the Jetson; `gemma3:4b` is the practical Gemma. (User has since pulled a Gemma 4 on Ollama for an accuracy reference — but mind the size for edge.)
- **Multi-frame helps MOTION threats, hurts OBJECT threats:** violence 25%→37% (caught a stabbing the single frame missed); weapons 28%→14% (a weapon is a single-frame object; extra frames + CoT made it over-reason). ⇒ **gate config should be per-rule, not one global setting.** Multi-frame ~tripled latency (7s→19s/call) — a real edge cost (acceptable since the gate runs infrequently).

### KEY DECISIONS
1. **Edge gate = local small VLM via Ollama** (no cloud). `gemma3:4b` leads. Pick by performance **on OUR task**, not generic benchmarks.
2. **Per-rule gate configuration** (multi-frame for concealment/violence, single-frame for weapons).
3. **Frame selection + curated labels are now the bottleneck** (see open problem below), not the harness.
4. The general VLM is the **V1 bridge**; the **Phase-2 trained action model** (on labeled motion data) is the real recall/edge fix.

### THE OPEN PROBLEM (current focus): frame selection for a frame-blind VLM
The VLM sees *frames*, not motion, so a single frame of a real theft/assault can look innocent (e.g., a robbery clip where the incriminating moment is them *yanking* a machine across frames). Our current picker chooses the **peak-detector-score** frame, which ≠ the frame where the threat is **visually obvious**. Recommended fixes (senior view): (a) **multi-frame spanning the event** — the biggest lever, already built; (b) anchor selection to the detector's **event moment** (state transition / motion peak / optical-flow spike), not raw score; (c) **crop to the tracked person** so the VLM focuses on the suspect's hands; (d) prefer a **temporal-capable small VLM** — note the team's own table put **SmolVLM2-2.2B #1 on Video & Temporal**, and our task is temporal; (e) ultimately a **trained temporal model** beats frame-picking. Cleanly measuring any of this needs a **curated, accurately-labeled eval set with more normals** (also produces the validated FPR for the registry).

### Where things are (for pickup)
- Branch **`unify-detector`** (off `theft-retail`, off `ayo/main`) holds: the unified `detector.py` (weapons+violence+theft+concealment+zones via `--zones`/`--concealment`), the local Ollama gate, the bake-off + toggles, the batch tool, and this doc. Pushed to `origin/unify-detector` (DEMILADE07).
- Standalone modules: `retail_zones.py`, `concealment.py`, `customization.py` (+ zone/concealment converters), `verification_gate.py` (mock/anthropic/openrouter/ollama, +cot, +multi-frame), `agent_mapper.py`.
- Configs: `configs/retail_pipeline_v1.json`, `all_threats_v1.json`, `retail_zones*.json`, `bytetrack_retail.yaml`. Tests: `tests/test_concealment.py`, `test_retail_zones.py`, `test_zone_customization.py` (all green). Big data (`data/anomaly/`, `*.dmg`, `runs/`) is gitignored.

## Checkpoint 2026-07-01 Frame-Selection Fixes — crop-to-person + event-moment (the cheap wins for the frame-blind VLM)
Built the two highest-ROI fixes for "the VLM sees frames, not motion" into the bake-off as A/B toggles (compose with the existing `--gate-frames`/`--cot`/`--use-agent-mapper`).
- **`--crop`** — sends the gate a tight CROP on the tracked suspect (bbox from pose tracking + `--crop-margin` padding) instead of the wide scene, so the VLM focuses on the hands / merchandise / weapon region. ROI per rule: concealment = the max-score person's box; violence = the union of people (interaction area); weapon = the weapon(s) + nearby people. The **Agent Mapper still gets the FULL frame** (it needs the whole environment). Verified: `theft_shop_01` crop = 204×142 on the shoplifter vs the full 640×360.
- **`--event-moment`** — anchors frame selection to the **peak MOTION frame** (frame-difference energy) within the detector-flagged window, not the peak detector-score frame — so it grabs the *action* moment (the reach/grab/yank), which is where a single frame is actually incriminating.
- Now **five independent gate toggles** to A/B: `--gate-frames N`, `--cot`, `--use-agent-mapper`, `--crop`, `--event-moment`. Each run's config prints in the table header. Verified end-to-end with `--models mock`; needs an Ollama run to measure effect on a curated set.
- **Still the gating dependency (unchanged): a curated, accurately-labeled eval set with real normals** — without it the toggle A/B stays noisy (concealment is already 100%, violence/weapon labels are approximate). That's the next real deliverable + it produces the validated FPR.
- Pushed to `origin/unify-detector`. — make the theft signal temporal (pose-sequence first, or video model), feeding off the `RetailZoneMonitor` shelf-interaction trigger + a rolling clip buffer → fused with the state machine → Verification Gate (upgraded to multi-frame). This is the piece that makes the product genuinely Veesion-like rather than a frame-guesser.

## Checkpoint 2026-07-04 Backend V1 Direction + Multi-Threat Clarification
Founder clarified the product thesis: the system is a **context-aware AI security intelligence layer**, not a single-purpose detector. The moat is that "threat" means different things to different customers, and even to different cameras within the same customer environment. The customer defines threat policy; the backend detects possible events; the verification layer confirms whether the event matches that configured threat.

### Strategic product understanding
- This should not become only a shoplifting app, a violence detector, or a VLM demo.
- The system should be a local camera-intelligence layer that can serve estates, retail shops, malls, offices, banks, warehouses, and later higher-friction sectors.
- Agent Mapper describes each camera's environment and zones. It must remain descriptive and should not decide threat policy.
- Customization Engine applies the user's threat definition through `user_config.json`.
- Detection Core produces cheap local signals such as people, weapons, pose, zones, dwell, concealment, violence, and later running/crowd/tampering/person-down.
- Verification Gate confirms or rejects a specific candidate alert using the rule and scene context.

### Important current backend gap
`CustomizationEngine.evaluate()` already returns multiple matching `CandidateAlert`s, sorted by priority, but the runtime paths usually select only:

```python
top_alert = candidate_alerts[0] if candidate_alerts else None
```

This means the architecture can represent multiple threats, but runtime behavior is still too close to "many signals -> one top alert." V1 needs "many signals -> multiple candidate alerts -> throttled verification queue -> multiple saved alert artifacts." This matters because shoplifting can occur in one part of a retail shop while armed robbery, violence, or panic happens in another.

### Always-on critical baseline decision
User configuration should not hide universal critical threats. The backend should separate:

```text
customer-specific threat policy
always-on critical safety baseline
```

The baseline should eventually include visible weapon/armed robbery, serious violence, person down/fall, fire/smoke, and camera tampering. Customer-specific rules then add business context such as shoplifting, loitering, vault-after-hours, gate tailgating, or power-outage + motion.

### Robbery should be a compound threat recipe
Robbery should not rely only on detecting a gun. Low-resolution CCTV often makes guns too small, blurry, or occluded. Robbery should be represented as a compound event using signals such as weapon candidate, violence, masked entry, counter rush, running/panic, person down, and crowd dispersal. The VLM gate then verifies the compound candidate, not just a tiny object box.

### VLM versus trained video models
The team discussed whether to rely on VLMs or fine-tune video models. Current conclusion: use a **hybrid architecture**.
- CV rules and trained/specialized video models should generate candidate events.
- VLMs should verify candidate events against the customer's specific rule and scene context.
- A VLM should not be expected to discover all threats from raw footage.
- True motion threats such as concealment, assault, stabbing, fall/person-down, running, fence climbing, and tampering are better candidates for fine-tuned temporal video models.

The immediate training strategy should not be "train a general threat model." It should be:

```text
choose one rule -> collect/label clips -> fine-tune a pretrained video model -> plug it into candidate generation -> VLM verifies -> measure FPR/recall
```

Most practical first targets: concealment/no-concealment, violence/no-violence, or person-down/fall.

### Data strategy clarification
Nigeria-specific data is not available at scale yet. The correct path is:
- use public/online/western CCTV data now for bootstrapping and initial fine-tunes;
- treat it as starting data, not final product data;
- collect event artifacts during pilots;
- require human review labels before using those artifacts for training;
- use the reviewed labels for active learning / supervised fine-tuning, not true reinforcement learning yet.

The feedback loop is:

```text
candidate event -> human review -> true threat / false alarm / missed threat / ambiguous -> curated dataset -> periodic supervised fine-tuning
```

### Current codebase status after repo review
What exists:
- `detector.py` unified branch can run weapons, violence, theft, zones, and concealment in one stream when the right flags are used.
- `customization.py` can convert assessments/zone/concealment outputs into RawEvents and evaluate all matching rules.
- `agent_mapper.py` can generate descriptive scene context.
- `verification_gate.py` supports mock, Anthropic, OpenRouter, local Ollama, multi-frame verification, CoT prompt mode, and artifact saving.
- `tools/gate_bakeoff.py` supports local VLM bakeoff with `--gate-frames`, `--cot`, `--use-agent-mapper`, `--crop`, and `--event-moment`.
- `tools/batch_anomaly.py` provides broad recall-style testing over anomaly clips.
- `retail_zones.py` and `concealment.py` are tested on synthetic logic.

What is missing:
- full GTM 12-rule V1 implementation;
- always-on critical baseline rules;
- multi-threat alert queue;
- runtime per-rule frame/clip selection;
- first-class robbery compound rule;
- validated FPR from a curated labeled eval set;
- true temporal video-model fine-tuning pipeline;
- human feedback/retraining queue;
- production TensorRT/export path.

### Ayo branch technical comparison as of 2026-07-04
Compared `unify-detector` with `ayo/main` at `4364ba4`.

Ayo's branch adds packaging/productization:
- installable `cvti` package;
- desktop app;
- build scripts and PyInstaller spec;
- local Ollama operational helpers in `cvti/verification/ollama.py`;
- default local model `gemma3:4b-it-qat`;
- offline VLM/user guide/software wiring docs.

Backend-intelligence features present in `unify-detector` but not Ayo's branch:
- general detector flags `--zones` and `--concealment`;
- general detector merges weapons, violence, theft, zones, and concealment into the same event stream;
- local gate provider named `ollama`;
- multi-frame `VerificationGate.verify()` that accepts a list of frames;
- CoT gate prompt mode;
- OpenRouter provider in this branch's gate;
- `tools/gate_bakeoff.py` with `--gate-frames`, `--cot`, `--use-agent-mapper`, `--crop`, `--event-moment`;
- `tools/batch_anomaly.py`;
- `configs/all_threats_v1.json`;
- `docs/GATE_MODEL_BAKEOFF.md`.

Ayo's general detector still lacks the unified zone/concealment flags and uses `mock`/`anthropic` gate choices in that path. Ayo's packaged `cvti/verification/gate.py` supports `local` and `openai_compatible`, but it is single-frame only and does not include CoT or the bakeoff frame-selection experiments. Both branches still need the top-alert pattern replaced with a multi-alert queue.

### New planning artifact
Created `plan.md` as the current backend V1 roadmap. It records:
- overview/product thesis;
- what's already built;
- what is ticked;
- what's missing;
- edge cases discussed;
- data/training strategy;
- minimum backend V1 definition;
- roadmap;
- technical comparison with Ayo's branch.

## Checkpoint 2026-07-09 VideoMAE Hybrid Temporal Signal Integrated

Added a standalone and runtime-ready video-action layer for the hybrid detector architecture.

### What changed

- Added `video_action_model.py`:
  - VideoMAE wrapper using `MCG-NJU/videomae-base-finetuned-kinetics`;
  - optional X3D wrapper for comparison, though the team is currently standardizing on VideoMAE because it gives more useful weak violence signals;
  - frame sampling helpers for single windows, beginning/middle/ending windows, and detector-centered event windows.
- Added `tools/video_action_probe.py`:
  - can run VideoMAE or X3D on a local clip;
  - can save the exact sampled frames;
  - can output JSON artifacts;
  - supports `--window-mode single`, `--window-mode segments`, and `--window-mode event`;
  - supports `--center-frame` for simulating "YOLO/pose found something suspicious at this frame."
- Added `video_action_hybrid.py`:
  - maps useful pretrained action labels into weak `RawEvent`s;
  - down-weights the raw video-model confidence before it reaches rules;
  - avoids treating irrelevant Kinetics labels as security truth.
- Added `video_action_runtime.py`:
  - keeps a rolling frame buffer in live detector runs;
  - when YOLO/pose/theft/concealment sees a suspicious moment, it samples frames around that moment and asks VideoMAE for weak temporal evidence.
- Added `configs/hybrid_video_action_v1.json`:
  - demo config for consuming weak `video_action` RawEvents.
- Added tests:
  - `tests/test_video_action_model.py`;
  - `tests/test_video_action_hybrid.py`;
  - `tests/test_video_action_runtime.py`.

### Live detector integration

`detector.py` now has optional flags:

```text
--video-action-backend {none,videomae,x3d}
--video-action-model
--video-action-window-seconds
--video-action-frames
--video-action-top-k
--video-action-cooldown
--video-action-device
```

The intended production path is:

```text
YOLO / pose / theft / concealment detects suspicious frame
-> current frame becomes event center
-> VideoMAE samples 16 frames around that moment
-> VideoMAE emits weak temporal labels
-> useful labels become down-weighted RawEvents
-> CustomizationEngine decides if the customer config cares
-> VerificationGate/VLM confirms or rejects
```

This confirms the desired hybrid system: VideoMAE is not the judge. It is a weak temporal witness. The config and VLM still decide final alert behavior.

### Validation so far

Focused tests pass:

```text
tests/test_video_action_model.py
tests/test_video_action_hybrid.py
tests/test_video_action_runtime.py
```

Smoke-tested detector runtime on `data/test_clips/violence_suspected.mp4`:

```text
[VideoAction] ... top=punching person (boxing)
[CONFIRMED] video_action_violence_candidate (MEDIUM)
```

VideoMAE local model cache is about 330 MB. X3D-S is about 30 MB, but X3D is not the current preferred backend because its outputs have been weaker on the tested threat clips.

### Current caveats

- Pretrained VideoMAE does not understand shoplifting/concealment directly. It is more useful for weak violence/motion evidence.
- The system still needs multi-alert queue semantics. The live detector still uses a top-alert pattern in the config/gate path.
- VideoMAE should stay optional and off by default until we validate runtime latency and false-positive behavior on a larger eval set.
- For shoplifting, the pose/concealment heuristic remains more relevant until we fine-tune a video model on concealment clips.
