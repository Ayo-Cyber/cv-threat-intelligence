# Argus — design brief

**For:** a full interface design of the Argus operator console
**Written:** 19 Aug 2026 · **Sources:** [AUDIT.md](AUDIT.md) §PD/§USR/§INV, and the shipped code

This brief exists because the interface grew one panel per feature and nobody
has stepped back to ask what people are trying to do with it. Everything below
is either a finding from the independent audit or a fact about what is actually
built today. Where a number appears, it was measured — see [NUMBERS.md](NUMBERS.md).

---

## 1. What the product is

Cheap computer-vision detectors watch CCTV cameras and over-fire constantly. A
vision-language model running **on the same machine** looks at what they flagged
and throws out the ones that aren't real. The operator gets the alerts that
matter, and no footage leaves the building.

Measured on held-out fire footage: raw detectors flag **90%** of ordinary clips
as a threat; with verification, **6.7%**, with no fires missed (n=39 clips,
95% CI on that recall is 70–100%).

It runs as a desktop app on a machine at the customer's site. Offline, no cloud.

## 2. Who actually uses it — and the central problem

The interface assumes the installer, the operator and the owner are **one
person**. In a real deployment they are three, with near-disjoint needs.

| | The guard / operator | The owner / buyer | The installer |
|---|---|---|---|
| **Frequency** | A whole shift, continuously | Almost never | Once, at install |
| **Where they are** | Walking the site, with a phone | An office, or their email | On a ladder with a laptop |
| **What they need** | *What needs me right now?* | *Was this worth the money?* | Cameras, zones, detectors working |
| **What they get today** | A flat list of alerts | **Nothing designed for them at all** | Config split across 4 panels |

**The buyer has no reason to ever open the app.** They signed the cheque, it
runs in a back office, and their only signal is whether staff complain. That is
the profile of a product that gets cancelled at renewal — not because it failed,
but because nobody could articulate what it did.

## 3. The three jobs, versus the nine surfaces

Navigation today: **Cameras · Alerts · Live · Map · Ask · Value · Learning ·
Rules · System**, plus ten detector toggles.

There are really three jobs:

| Job | How often | Currently |
|---|---|---|
| **Watch** — is everything OK right now? | Continuous | Split across Live, Cameras, Map |
| **Triage** — what needs me, what did I do about it? | Per incident | Alerts — a flat list |
| **Configure** — cameras, zones, rules, detectors | Once, at install | Split across Rules, System, Cameras, toggles |

The job that happens **once** occupies about half the navigation permanently.
The job that happens **constantly** gets one panel.

This is a reorganisation, not a rewrite — the underlying panels mostly work.

## 4. The single largest product gap: there is no triage workflow

Sorting is not triage. Colour is not triage.

The model cut alerts by **86%**. The remaining 14% still land in an
undifferentiated list. At 20 cameras that is still dozens per shift.

**So the product's core claim — that it reduces alert fatigue — is currently
delivered entirely by the model and not at all by the interface.**

A guard at 2am needs to know:

- **What needs me right now?** One unambiguous next item, not a list.
- **Who is already on this?** There is no ownership concept, so with two guards
  on shift both respond or neither does.
- **What happened last shift?** No handover surface; context resets every shift.
- **What did I decide, and can I show it later?** Labels exist for model
  training, but there is no incident record a manager can review.

Needed: explicit alert states (`NEW → ACKNOWLEDGED (by whom, when) → RESOLVED
(outcome, note)`), ownership visible to everyone, a "needs attention now" view
defaulting to one alert at a time, a shift handover summary, and a per-incident
record that exports as a PDF.

## 5. The response loop has a walk-to-the-office step in it

Alerts already reach a phone by Telegram. **Every response action requires the
desktop app** — reviewing frames, acknowledging, labelling.

The notification is mobile; the response is not. That is backwards for a job
whose defining feature is *walking around*. From the guard's seat this is one
experience: *"I get told about things I can't do anything about from where I am
standing."* If a shift feels like that, the guard stops trusting the alerts —
and a monitoring product the monitor ignores has zero value regardless of its
precision.

The cheap first version is **not an app**. It is a mobile-responsive web view
served over the local network, showing the alert with its frames, an
Acknowledge button, and a note field. Telegram deep-links into it.

*(It must be authenticated — this endpoint is on a customer's network. Auth now
exists; see §7.)*

## 6. Silence is the system's characteristic failure

This shows up three separate times in the audit, and it is worth treating as one
design principle:

- A quiet night and a dead engine look identical.
- An offline camera and a camera watching an empty corridor look identical.
- A detector that correctly found nothing and one that has thrown on every frame
  for a week look identical.

Traditional alarm panels solved this decades ago with a blinking light. The
interface needs a continuously visible liveness signal — per camera, with
last-frame-processed time — and a daily "all systems normal" message rather than
only ever speaking when something is wrong.

Much of the plumbing for this now exists (§7); almost none of it is designed.

## 7. What is actually built — including things with no interface yet

This matters: several capabilities landed in the last week and have **no UI at
all**. A design that ignores them will miss half the product.

### Screens that exist

| Screen | State |
|---|---|
| Cameras | Works. Now shows real link state (live / reconnecting / **OFFLINE** / not monitoring). |
| Alerts | Flat list + detail panel with evidence frames, a clip cine-loop, the model's reasoning, and Acknowledge / True / False. |
| Live | Multi-camera wall with live boxes. |
| Map | Site map with pins that light on alert. |
| Ask | Natural-language query over cameras. Audit calls it a demo feature, not a product feature. |
| Value | New. Raw-vs-shown comparison, incidents, false alarms prevented, attention-hours saved, site-configurable rates. |
| Learning | Operator labels → calibration; noisy rules get demoted. |
| Rules | Ten detector toggles per camera, grouped Security / Safety. |
| System | Site name, notifications, gate health, per-component error counters, data retention, diagnostics download. |

### Built, working, and **completely undesigned**

| Capability | Status |
|---|---|
| **Sign-in and first-run account creation** | Backend enforced. **No login screen exists.** The app is currently unusable once auth merges. |
| **Three roles** (Owner / Operator / Installer) | Enforced server-side on 25 methods. No role-aware navigation. |
| **Audit trail** — append-only, tamper-evident | Owner-only. No viewer. |
| **User management** | Add / remove / change role. No screen. |
| **Disk-encryption status** | Probed per platform. Nowhere to show it. |
| **Unverified alerts** | When verification fails the alert now reaches the operator marked `UNVERIFIED — TrueSight could not decide`. **Needs a distinct visual treatment — it is not a detection.** |
| **Legal hold** | Per alert, exempts evidence from the retention purge. Minimal UI. |
| **Evidence export** | Per incident, zipped. Button only. |

## 8. Constraints — real, not preferences

- **Desktop app**, PyQt6 + QtWebEngine. The UI is a **single `index.html`** —
  inline CSS and JS, no build step, no framework, no CDN. Roughly 1,500 lines today.
- **Fully offline.** No web fonts, no external assets, no analytics.
- **Runs on modest site hardware** next to a detection engine using ~6 GB of RAM.
- **Dark and light themes** both exist and must keep working.
- The **mobile view** is served by the same machine over the local network.
- Alerts carry real evidence: several JPEG frames, an annotated subject shot, a
  short clip, a confidence score, and a sentence of model reasoning.
- **Latency is ~28 s median** from detection to verified alert. The interface has
  to make that wait legible rather than hide it.

## 9. What we want from the design

1. **An information architecture built on the three jobs**, not the nine panels.
   Configuration behind a settings surface. Watch and Map probably merge — they
   answer the same question at different zoom levels.
2. **A real triage workflow**: next-action-first, ownership, states, handover,
   incident record.
3. **A mobile response view** — alert, frames, acknowledge, label, note.
4. **An owner surface**: a weekly summary they receive without asking, in
   business terms, every figure traceable to real events.
5. **Role-aware navigation** for three roles that already exist in the backend.
6. **A first-run flow**: create the owner account → add cameras → zones → pick a
   use case → confirm detectors → send a test alert. Under 15 minutes, by
   someone non-technical.
7. **A liveness language** — the blinking-light problem — covering camera state,
   engine state, degraded components, and unverified alerts.
8. **Honest states designed, not bolted on**: `UNVERIFIED`, camera `OFFLINE`,
   component degraded, no verification history yet, disk not encrypted. The
   product's credibility rests on saying what it does not know.

## 10. What would make the design wrong

- Treating the owner as a power user. They will open it monthly at most.
- A triage flow that assumes a mouse and a large screen. Half the job is on a phone.
- Hiding the 28-second verification wait instead of designing for it.
- Showing an operator anything they cannot act on.
- Making `UNVERIFIED` look like a detection. It is the system saying *I don't know*.
- Marketing numbers without their sample size. Every published figure carries n
  and a confidence interval, deliberately, and the interface should hold that line.
