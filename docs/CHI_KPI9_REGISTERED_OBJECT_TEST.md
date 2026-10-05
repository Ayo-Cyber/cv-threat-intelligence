# Registered Object Pilot: Electron Test

## What Is Connected

Registered objects in camera details uses the live API and shared camera engine,
not demo detections. A timed appearance change is sent to the existing verifier
with reference/current evidence. A verified change is not proof of theft.

## Operator Test

1. Use a fixed camera with a clearly visible, textured object. Keep the camera still.
2. Start monitoring. Open Cameras, select Scene review for the camera, then
   Registered objects.
3. Capture view, drag a tight rectangle around the object, name it and register.
4. Keep people out of the region until the state becomes present. Registration
   waits for three stable, unobstructed frames; an empty scene must not be registered
   as an object reference.
5. First walk past the object without moving it. No change incident should appear.
6. Remove the object, then move clear. With the default setting, sustained visible
   change takes eight seconds to become a candidate. VLM processing and queue time
   are additional; no fixed alert-latency promise is made.
7. Check the fresh incident and reference/current panel. Record rejected/no-alert
   cases too. An old incident is not a result for this test.
8. Restore the object. Use Recapture only after confirming it is present and clear.
   Engine restart or a long interruption also requires recapture in this pilot.

The registered-object row reports detector state, not the final VLM verdict.
Removal stops future monitoring; historical incidents/references are retained.

## Completed Checks

- Focused backend, permissions, API contract, lifecycle and existing serving tests.
- Runtime-to-queue evidence and controlled-verifier-to-SQLite plumbing tests.
- Desktop/mobile region-coordinate tests and actual Electron registration/removal.
- Production renderer and Electron compilation.

These do not replace real-camera accuracy testing. Pipeline B automatic discovery,
movement classification, automatic restoration and camera-jitter alignment are
not part of this initial registered-region pilot.
