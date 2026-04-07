# MVP POC Plan

## Objective
Deliver a believable first-phase prototype in less than 36 hours.

The prototype should prove that:
- a live camera feed can be ingested
- a vision model can run on the feed
- configured detections can trigger an alert
- evidence can be saved for later review

## What Counts As Success
The POC is successful if you can show your co-founder:
- a webcam feed running through the app
- detections visible on screen
- a threat banner appearing for configured classes
- saved event frames or clips in `runs/detect`
- the same script working with a video file
- the same script being adaptable to `RTSP`

## What Not To Solve In This First Window
Do not block the POC on:
- Jetson deployment
- cloud production setup
- multi-camera orchestration
- mobile apps
- dashboards
- custom action-recognition architecture

## Recommended Scope
### Track A: pipeline proof
- webcam input
- pretrained or custom YOLO weights
- threat classes configured by command line
- on-screen rendering
- evidence saving

### Track B: backup demo
- prerecorded demo video
- same detector pipeline
- stable presentation path if live webcam performance is inconsistent

## Recommended Threat Prioritization
### Best first target
- `person`
- `knife`
- `gun`

### Higher difficulty
- `fight`
- `steal`
- bag snatching on motorcycle

The first group is more realistic for this deadline because they can be approached as object detection. The second group is more temporal and behavior-driven, so they usually need more data and a more specialized pipeline.

## Execution Sequence
### Step 1
Run the pipeline immediately with webcam + pretrained weights.

Suggested command:
```powershell
python detector.py --source 0 --weights yolov8n.pt --threat-classes person --show
```

### Step 2
Verify these basics:
- the webcam opens
- the model loads
- detections render correctly
- events save under `runs/detect`

### Step 3
Test with a prerecorded video to create a stable fallback demo.

### Step 4
When an RTSP camera is available, test the exact same script with the stream URL.

### Step 5
If you already have or can obtain specialized weights, swap them into the same pipeline.

## Colab Recommendation
Use Colab only for:
- trying training notebooks
- testing fine-tuning ideas
- exporting custom weights

Do not depend on Colab for the live POC demo unless absolutely necessary.

## Deployment Recommendation
Do not attempt Jetson or Jetson-like simulation in this first phase unless the local POC is already complete and stable.

For this first deadline, the most important question is:
"Can the pipeline detect anything meaningful live?"

That answer matters more than edge deployment right now.

## Immediate Work Checklist
- install dependencies
- run webcam smoke test
- verify saved evidence
- collect a short demo video
- test a few staged scenes
- tune thresholds
- prepare a backup recorded demo

## Next Phase After This POC
Once the first POC works, the next build phase should focus on:
- RTSP-first camera testing
- custom dataset collection
- Nigeria-specific scenarios
- custom model training or fine-tuning
- deployment benchmarking on practical hardware
