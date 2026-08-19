# Argus — what we have measured

_Every figure below comes from `python -m cvti.eval` on **held-out** clips the models never trained on, verified by the local VLM (gemma3:4b) on one laptop. Regenerate with `python tools/make_numbers_sheet.py`._

## Headline

On fire detection, the cheap computer-vision detector alone flags **90.0% of ordinary footage** as a threat — unusable on its own. Local AI verification cuts that to **6.7%** while missing **0 of 9 fires**.

That gap is the product.

## Measured results

| Threat | Clips | | Precision | Recall | False alarms | Alerts shown |
|---|---:|---|---:|---:|---:|---:|
| **Fire / smoke** | 39 (9+/30−) | detector alone | 25.0% | 100.0% | 90.0% | 39 |
| | | **+ verification** | **81.8%** | **100.0%** | **6.7%** | **12** |
| **Crowd forming** | 38 (8+/30−) | detector alone | 38.9% | 87.5% | 36.7% | 52 |
| | | **+ verification** | **60.0%** | **75.0%** | **13.3%** | **14** |
| **Theft (balanced)** | 36 (9+/27−) | detector alone | 37.5% | 100.0% | 55.6% | 201 |
| | | **+ verification** | **53.3%** | **88.9%** | **25.9%** | **61** |
| **Theft (strict)** | 36 (9+/27−) | detector alone | 37.5% | 100.0% | 55.6% | 201 |
| | | **+ verification** | **63.6%** | **77.8%** | **14.8%** | **28** |
| **Theft (before tuning)** | 36 (9+/27−) | detector alone | 37.5% | 100.0% | 55.6% | 201 |
| | | **+ verification** | **37.5%** | **100.0%** | **55.6%** | **191** |

_“False alarms” = share of normal clips that raised an alert. “Alerts shown” = what an operator would actually see._

## Sensitivity is a measured setting, not a claim

Theft strictness trades recall for precision, so it is the operator's choice — with the cost of each option measured:

| Setting | Catches | Precision | False alarms |
|---|---:|---:|---:|
| `balanced` (default) | 88.9% | 53.3% | 25.9% |
| `strict` | 77.8% | 63.6% | 14.8% |

Default is `balanced`: for security, a missed threat costs more than a reviewed false alarm.

## Coverage: what is measured, and what is not

| Capability | Status |
|---|---|
| Fire / smoke | ✅ measured — 100.0% caught, 6.7% false alarms |
| Crowd forming | ✅ measured — 75.0% caught, 13.3% false alarms |
| Theft / concealment | ✅ measured — 88.9% caught, 25.9% false alarms |
| Panic running | ⚠️ built and demonstrable, **not yet validated** |
| Person collapsed | ⚠️ built and demonstrable, **not yet validated** |
| Weapons | ⚠️ built and demonstrable, **not yet validated** |
| Violence / assault | ⚠️ built and demonstrable, **not yet validated** |
| Camera tampering | ⚠️ built and demonstrable, **not yet validated** |
| Loitering (zone dwell) | ⚠️ built and demonstrable, **not yet validated** |
| Custom rules in plain English | ⚠️ built and demonstrable, **not yet validated** |


The blocker is labelled test footage, not the detectors — they are deterministic rules, so nothing needs training. Fire and crowd were measurable because raw footage of them is easy to find; searching for falls and fights mostly returns news coverage OF incidents, so those need a proper labelled set (RWF-2000, UR Fall) rather than more searching.

## Runs on one machine

Measured on a single MacBook Pro (18 GB), 5 cameras, detection and verification both local — nothing left the machine:

| | Measured |
|---|---|
| Cameras | 5 concurrent |
| Per-camera rate | 6.2 fps sustained (above the 4 fps target) |
| Detector cost | 163 ms for a batch of 5 cameras |
| Alert latency (detected → verified) | median 28 s, best 11 s |
| Memory | ~3 GB engine + ~3 GB local model |

The detector is not the limit — it has headroom at 5 cameras. Latency comes from the verification model, and scales with the number of workers: two workers cut median latency from 46.5 s to 28 s at no extra memory cost, so worker count now derives from camera count automatically.

## Caveats we volunteer

- **Sample size.** 36–39 clips per threat. Directionally honest, not an SLA.
- **One model.** All figures use gemma3:4b locally; a larger gate model would likely score higher but costs more per verdict.
- **Clip-level scoring.** One alert on a threat clip counts as a catch, which is how an operator experiences it.
- **Labels are hand-checked.** Of the first 12 clips a search returned for “fire”, only 3 contained fire — the rest were news segments and logos. Every clip in these sets was eyeballed; rejects are kept aside with the reason.
- **Latency is verification, not detection.** Detection is ~instant; the median 28 s is the local model reasoning about the frames. A cloud model would be faster but would mean sending footage off-site.

## If asked “how accurate is it?”

> Fire is measured on held-out footage: we catch every fire in the set and cut false alarms from 90% to under 7%. Theft is measured too — 89% caught, with verification removing about 70% of the noise. Panic and crowd are built and you will see them run, but we are still validating them, so I will not quote a number I cannot defend yet.

