# Data retention

*Customer-facing. Written to be shown to a data protection officer, a
procurement questionnaire, or the person whose image is in the footage.*

## What Argus stores

| Data | Where | Why |
|---|---|---|
| Camera images and short clips of a confirmed alert | `<output>/events/<timestamp>_<camera>_<rule>/` | So a human can judge whether the alert was real |
| An event record — time, camera, rule, confidence, the model's stated reason | `<output>/events.db` | The reviewable alert list, and the audit trail of what was decided |
| Daily suppression counts | `<output>/events.db` | To show what the system filtered out |
| Application logs | `<output>/logs/` | Support and diagnosis. **No images.** |

Everything stays on the customer's machine. Nothing is uploaded, and
verification runs on-device.

## How long

**Evidence is deleted 30 days after it was recorded**, by default. The site
owner can change this in **System → Data retention**; the minimum is 1 day.

A purge deletes the frames, the clips **and** the database record together. It
runs hourly, and again immediately when disk usage crosses the critical
threshold.

## What is deliberately kept longer

Blind time-based deletion would destroy the exact records a customer needs, so
two categories survive their own expiry:

1. **Anything on legal hold.** An operator marks an alert on the alert screen —
   for an insurance claim, a police report, a dispute. It is kept until the hold
   is released, and the System panel shows how many such items exist.
2. **Anything not yet reviewed.** Nobody has decided what it was. Deleting an
   open incident on a timer destroys the record while the question is still
   live. Labelling it (True threat / False alarm / Acknowledge) settles it, and
   it then expires normally.

Both are counted and shown in **System → Data retention**, so "why is this still
here?" always has an answer.

## Erasure and export

- **Erasure request:** delete the specific event from the alert list, or shorten
  the retention period and let the next purge run. Frames, clips and the record
  go together — there is no copy left behind. An orphan sweep also removes any
  evidence directory with no corresponding record.
- **Export before expiry:** *Export evidence* on any alert produces a zip of
  that incident's frames and clips. Use it before evidence expires if it is
  needed for a claim. The export carries a notice that it contains images of
  identifiable people.
- **Support bundles are not evidence.** *Download diagnostics* deliberately
  contains **no** images, video, or event records — only logs and aggregate
  counts, listed in its own manifest.

## Disk

Retention also protects the recording itself. An edge machine with no purge
fills its disk, and when it does, writes fail and evidence stops being recorded
at the moment it is most needed. Argus warns at 85% and, at 95%, purges
oldest-first — still refusing to touch anything on legal hold or unreviewed.

If the disk is critical and nothing is deletable because everything is held,
Argus logs that explicitly rather than deleting a held item or silently filling
up.

## What this does not cover

Argus is an assistive layer that reduces operator load. It is not a guarantee of
detection, and its measured recall is published in
[NUMBERS.md](NUMBERS.md) with sample sizes and confidence intervals.
Retention policy is a technical control; it is not legal advice, and a
deployment still needs its own privacy notice, DPIA and signage.
