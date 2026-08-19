# Argus — what we have measured

_Every figure below comes from `python -m cvti.eval` on **held-out** clips the models never trained on, verified by the local VLM (gemma3:4b) on one laptop. Regenerate with `python tools/make_numbers_sheet.py`._

## Headline

On fire detection, the cheap computer-vision detector alone flags **90.0% of ordinary footage** as a threat — unusable on its own. Local AI verification cuts that to **6.7%** (95% CI 1.8%–21.3%, n=30 normal clips) while missing **0 of 9 fires**.

Stated the way it should be stated: **100.0% recall on 9 held-out positive clips (95% CI 70.1%–100.0%)**. 9 clips is a small sample and the interval says so — the lower bound, not the point estimate, is what we will defend.

That gap is the product.

## Measured results

| Threat | Clips | | Precision | Recall | False alarms | Alerts shown |
|---|---:|---|---:|---:|---:|---:|
| **Fire / smoke** | 39 (9+/30−) | detector alone | 25.0%<br><sub>n=36, CI 13.8%–41.1%</sub> | 100.0%<br><sub>n=9, CI 70.1%–100.0%</sub> | 90.0%<br><sub>n=30, CI 74.4%–96.5%</sub> | 39 |
| | | **+ verification** | **81.8%**<br><sub>n=11, CI 52.3%–94.9%</sub> | **100.0%**<br><sub>n=9, CI 70.1%–100.0%</sub> | **6.7%**<br><sub>n=30, CI 1.8%–21.3%</sub> | **12** |
| **Crowd forming** | 38 (8+/30−) | detector alone | 38.9%<br><sub>n=18, CI 20.3%–61.4%</sub> | 87.5%<br><sub>n=8, CI 52.9%–97.8%</sub> | 36.7%<br><sub>n=30, CI 21.9%–54.5%</sub> | 52 |
| | | **+ verification** | **60.0%**<br><sub>n=10, CI 31.3%–83.2%</sub> | **75.0%**<br><sub>n=8, CI 40.9%–92.9%</sub> | **13.3%**<br><sub>n=30, CI 5.3%–29.7%</sub> | **14** |
| **Theft (balanced)** | 36 (9+/27−) | detector alone | 37.5%<br><sub>n=24, CI 21.2%–57.3%</sub> | 100.0%<br><sub>n=9, CI 70.1%–100.0%</sub> | 55.6%<br><sub>n=27, CI 37.3%–72.4%</sub> | 201 |
| | | **+ verification** | **53.3%**<br><sub>n=15, CI 30.1%–75.2%</sub> | **88.9%**<br><sub>n=9, CI 56.5%–98.0%</sub> | **25.9%**<br><sub>n=27, CI 13.2%–44.7%</sub> | **61** |
| **Theft (strict)** | 36 (9+/27−) | detector alone | 37.5%<br><sub>n=24, CI 21.2%–57.3%</sub> | 100.0%<br><sub>n=9, CI 70.1%–100.0%</sub> | 55.6%<br><sub>n=27, CI 37.3%–72.4%</sub> | 201 |
| | | **+ verification** | **63.6%**<br><sub>n=11, CI 35.4%–84.8%</sub> | **77.8%**<br><sub>n=9, CI 45.3%–93.7%</sub> | **14.8%**<br><sub>n=27, CI 5.9%–32.5%</sub> | **28** |
| **Theft (before tuning)** | 36 (9+/27−) | detector alone | 37.5%<br><sub>n=24, CI 21.2%–57.3%</sub> | 100.0%<br><sub>n=9, CI 70.1%–100.0%</sub> | 55.6%<br><sub>n=27, CI 37.3%–72.4%</sub> | 201 |
| | | **+ verification** | **37.5%**<br><sub>n=24, CI 21.2%–57.3%</sub> | **100.0%**<br><sub>n=9, CI 70.1%–100.0%</sub> | **55.6%**<br><sub>n=27, CI 37.3%–72.4%</sub> | **191** |

_“False alarms” = share of normal clips that raised an alert. “Alerts shown” = what an operator would actually see. Every rate carries its denominator and a 95% Wilson score interval — at these sample sizes the point estimate on its own would overstate what we know._

## Sensitivity is a measured setting, not a claim

Theft strictness trades recall for precision, so it is the operator's choice — with the cost of each option measured:

| Setting | Catches | Precision | False alarms |
|---|---:|---:|---:|
| `balanced` (default) | 88.9%<br><sub>n=9, CI 56.5%–98.0%</sub> | 53.3%<br><sub>n=15, CI 30.1%–75.2%</sub> | 25.9%<br><sub>n=27, CI 13.2%–44.7%</sub> |
| `strict` | 77.8%<br><sub>n=9, CI 45.3%–93.7%</sub> | 63.6%<br><sub>n=11, CI 35.4%–84.8%</sub> | 14.8%<br><sub>n=27, CI 5.9%–32.5%</sub> |

Default is `balanced`: for security, a missed threat costs more than a reviewed false alarm.

## Coverage: what is measured, and what is not

| Capability | Status |
|---|---|
| Fire / smoke | ✅ measured — 100.0% caught (n=9, CI 70.1%–100.0%), 6.7% false alarms |
| Crowd forming | ✅ measured — 75.0% caught (n=8, CI 40.9%–92.9%), 13.3% false alarms |
| Theft / concealment | ✅ measured — 88.9% caught (n=9, CI 56.5%–98.0%), 25.9% false alarms |
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

- **Sample size.** 36–39 clips per threat, so every rate above carries its denominator and a 95% Wilson interval. Those intervals are wide on purpose — they are what the sample supports. Directionally honest, not an SLA.
- **One model.** All figures use gemma3:4b locally; a larger gate model would likely score higher but costs more per verdict.
- **Clip-level scoring.** One alert on a threat clip counts as a catch, which is how an operator experiences it.
- **Labels are hand-checked.** Of the first 12 clips a search returned for “fire”, only 3 contained fire — the rest were news segments and logos. Every clip in these sets was eyeballed; rejects are kept aside with the reason.
- **Latency is verification, not detection.** Detection is ~instant; the median 28 s is the local model reasoning about the frames. A cloud model would be faster but would mean sending footage off-site.

## If asked “how accurate is it?”

> Fire is measured on held-out footage: we caught all 9 fires in the set and cut false alarms from 90% to under 7%. It is 9 clips, so the honest floor on that recall is 70.1%, not 100% — I will quote you the interval, not the headline. Theft is measured too: 89% caught, verification removing about 70% of the noise. Panic and falls are built and you will see them run, but they are not validated yet, so I will not quote a number I cannot defend.

