# Argus Pilot Agreement — Template

*EP-08-T4 (INV-01). Fill the blanks with the customer BEFORE deployment and
have both parties sign. A pilot without success criteria agreed in advance
produces an ambiguous outcome and no case study. Legal terms (DPA/ToS) come
from the EP-02-T1 legal review — attach them; this document is the
operational agreement.*

---

**Pilot site:** ______________________ ("Customer")
**Provider:** ______________________ ("Argus")
**Duration:** 30 days, from ____ / ____ / ______ to ____ / ____ / ______
**Review points:** day 7 (installation + tuning review), day 15 (mid-pilot),
day 30 (outcome review against the criteria below).

## 1. What is deployed

One Argus edge machine on the Customer's premises, connected to
______ cameras. All video processing happens on that machine. Footage never
leaves the premises; alert notifications go only to the channels the Customer
configures. Remote health monitoring (heartbeat) is **off unless** the
Customer opts in below — and sends only the documented status fields
([HEARTBEAT.md](HEARTBEAT.md)), never images.

Heartbeat opt-in: ☐ yes ☐ no    Notification channel: ______________

## 2. Data use, retention, liability

- Evidence (short clips + frames of confirmed alerts) is stored on the edge
  machine only, retained **______ days** (default 30), then deleted
  automatically. Items placed on legal hold or still under review are kept.
  ([DATA_RETENTION.md](DATA_RETENTION.md) is the full policy — DPO-ready.)
- The Customer remains data controller; Argus is processor per the attached
  DPA. ☐ DPA attached
- Argus is a detection *aid*. It does not replace the Customer's security
  procedures, and the Provider's liability is limited per the attached ToS.
  ☐ ToS attached
- Accounts are created by the Customer's owner user; the Provider holds no
  credentials to the Customer's system.

## 3. Success criteria — agreed in advance

*Measured by the product's own published figures (Value screen / weekly
summaries — every figure is a row count over the event store).*

| # | Criterion | Target (fill in) | Measured by |
|---|---|---|---|
| 1 | Monitoring uptime over the 30 days | ≥ ____ % | daily assurance + health history |
| 2 | Alerts shown to staff per day (noise) | ≤ ____ /day | weekly summary "shown" |
| 3 | Confirmation rate (real / shown) | ≥ ____ % | triage outcomes |
| 4 | Missed incidents known to staff | ≤ ____ | Customer's log vs event store |
| 5 | Operator satisfaction (guards using it) | ≥ ____ / 5 | day-30 survey, same 5 questions as baseline |
| 6 | (site-specific) ______________________ | ____ | ____ |

## 4. Baseline — captured BEFORE deployment

*Without this, improvement cannot be demonstrated. Fill in during the week
before installation — see [PILOT_BASELINE.md](PILOT_BASELINE.md) worksheet.*

- Current false-alarm burden: ______ alarms/week, ~______ staff-minutes each
- Incidents in the last 90 days (thefts, incidents requiring response): ______
- Current review practice (who watches, when): ______________________
- Baseline operator survey completed: ☐ (attach)

## 5. Reference

If the day-30 review meets criteria 1–5, the Customer agrees to act as a
reference: ☐ named case study ☐ anonymous case study ☐ reference calls only.

## 6. Signatures

Customer: ______________________  Date: ________
Provider: ______________________  Date: ________
