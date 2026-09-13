# Chi Pilot Pocketing And Bagging Design

## Goal

Strengthen the existing pose-based concealment path so Chi pilot scenario 10 can reliably generate candidates for pocketing or placing a product into a personal bag without treating normal browsing or trolley use as theft.

## Existing Foundation

`cvti/retail/concealment.py` already scores a short per-person sequence using destination proximity, reach-and-retract motion, and dwell at the waist or a personal bag. It deliberately excludes shopping trolleys and baskets. The Customization Engine and Agent Mapper restrict the meaning to compatible retail or merchandise contexts, and TrueSight makes the final judgment from temporal evidence.

Two production integration defects must be corrected before tuning:

1. Pose inference runs on a configurable heavy stride. A skipped pose frame currently reaches `ConcealmentDetector.update()` as an empty observation, which deletes the track history. Skipping computation must not mean the person left.
2. The multi-camera serving path never passes personal-bag detections to `ConcealmentDetector`, so its bag destination branch is inactive there.

## Temporal State Contract

Add an explicit distinction between a sampled pose observation and an unsampled video frame:

- Call `ConcealmentDetector.update()` only when pose inference actually ran.
- Add `expire(timestamp)` to remove tracks whose last sampled pose exceeds a configurable grace period.
- A sampled pose result that genuinely omits a previously seen person starts the grace period but does not erase the sequence immediately.
- Empty scenes still release all state after the grace period, preventing per-visitor memory growth.

This preserves the performance benefit of `heavy_stride` while allowing four above-threshold sampled observations to accumulate.

## Personal-Bag Grounding

Use the existing shared COCO YOLO result; do not add another model pass. Extract class IDs 24, 26, and 28 (`backpack`, `handbag`, `suitcase`) before person-only tracking.

Associate each bag with a pose track only when the bag overlaps that person's expanded box or is within a body-scale distance. `ConcealmentDetector` receives bags by track so one shopper's bag cannot become every shopper's destination. Preserve the standalone detector's existing global `bag_bboxes` interface for backward compatibility, but the production path uses track-specific associations.

## Candidate Meaning

A candidate remains a temporal witness, not a theft verdict. It records:

- destination: `waist` or `bag`
- reach, retract, destination, and dwell component scores
- track ID and subject box
- whether evidence was limited by missing joints
- associated bag box and label when present

The detector does not claim to identify the exact product. Scene context and merchandise zones increase plausibility, while TrueSight sees three chronological full frames plus the existing subject crop and decides whether the sequence visibly supports pocketing or bagging merchandise.

The verification prompt must explicitly reject ordinary browsing, holding a phone, adjusting clothing, carrying an item openly, and placing goods in a trolley or shopping basket.

## Tuning Policy

Do not lower thresholds globally before evaluation. First repair history and bag wiring, then measure the existing `0.63` score threshold and four-frame persistence on Chi clips. Any threshold change must be backed by labeled positive and negative intervals and recorded in the evaluation artifact.

## Testing And Acceptance

Automated tests reproduce the stride regression and prove the same motion fires with `heavy_stride=1` and `heavy_stride=2`. Additional tests cover stale expiry, personal-bag extraction, per-person bag association, two people with only one bag, trolley exclusion, normal browsing, clothing adjustment, missing hips, configuration plumbing, candidate metadata, and three-frame gate evidence.

Pilot evaluation includes pocket-positive, bag-positive, trolley-safe, phone-to-pocket, clothing-adjustment, ordinary browsing, and open-carry clips. Report candidate recall, post-gate precision/recall, duplicates per incident, detection delay, and pose-stage throughput. A passing pilot build must demonstrate at least one true pocket sequence and one true bag sequence end to end through rules, Agent Mapper context, TrueSight, persistence, and UI evidence.

## Non-Goals

- Product SKU recognition; that belongs to Chi scenario 3.
- Shelf inventory reconciliation or removed-object persistence; that belongs to scenario 9.
- Training or fine-tuning a new action model during this pilot slice.
- Treating the detector score as the final theft judgment.
