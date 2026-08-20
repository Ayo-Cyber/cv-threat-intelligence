# Security

How Argus is secured, what it does not defend against, and the answers to the
questions procurement asks. Written to be read by someone deciding whether to
put this on their network.

Vulnerability reports: open a GitHub security advisory on this repository.

---

## What the system holds

| Data | Where | Sensitivity |
|---|---|---|
| Camera images and clips of confirmed alerts | `<output>/events/` | Identifiable people |
| Event records — time, camera, rule, the model's stated reason | `<output>/events.db` | Identifiable behaviour |
| Credentials | `<output>/auth.db`, mode `0600` | Password hashes, never plaintext |
| Audit trail | `<output>/audit.db`, mode `0600` | Who did what, hash-chained |
| Application logs | `<output>/logs/` | No images |

Verification runs on-device. Nothing is uploaded; the system works with no
internet connection at all.

## Identity and access

**Accounts.** No default account and no default password ship with the product —
the first owner account is created during setup, so there is no known credential
to be found in a forum post. Passwords are hashed with `scrypt` where the Python
build provides it and PBKDF2-HMAC-SHA256 at 600,000 rounds otherwise; each hash
records which produced it. Passwords are never stored reversibly.

**Login.** An unknown username and a wrong password produce the same message and
the same work, so the endpoint cannot be used to enumerate users. Five failures
lock an account for five minutes; every attempt is logged. Sessions expire
(8 hours by default, configurable) and are revoked when a password changes.

**Roles.** Three, enforced in the backend rather than by hiding controls:

| | Owner | Operator | Installer |
|---|:--:|:--:|:--:|
| View alerts and evidence | ✅ | ✅ | — |
| Review / label alerts | ✅ | ✅ | — |
| Live camera view | ✅ | ✅ | ✅ |
| Configure cameras and zones | ✅ | — | ✅ |
| Configure detectors | ✅ | — | ✅ |
| Site settings, retention | ✅ | — | — |
| Start / stop monitoring | ✅ | — | ✅ |
| Export evidence | ✅ | ✅ | — |
| Legal hold | ✅ | ✅ | — |
| Audit trail | ✅ | — | — |
| Manage users | ✅ | — | — |

An operator cannot disable a detector. An installer cannot read recorded
incidents. Both are enforced server-side and covered by tests that call the
backend directly, bypassing the interface.

**Camera frame endpoints.** Every route on both frame servers requires a
per-run capability token, supplied as a header or query parameter, compared in
constant time. They bind to `127.0.0.1` and set no CORS header — an
unauthenticated frame endpoint would put live camera feeds on an open port.

## Audit trail

Append-only and tamper-evident, and these are different properties.

*Append-only*: no update or delete exists in the module, and SQLite triggers
refuse both even against direct SQL.

*Tamper-evident*: each entry hashes its own contents plus the previous entry's
hash. Editing a past entry breaks every hash after it, and `verify()` names the
first broken row. This does not prevent someone with the disk from rewriting the
whole chain — it makes a partial edit, which is the realistic attack, detectable.

Captured: logins, footage access, configuration and detector changes, alert
resolutions, evidence export, purges, and role changes — each with actor,
timestamp and target. Readable and exportable by the Owner role only, and stored
in its own database so a single deletion cannot remove both the footage and the
record of who touched it.

## Data at rest

**Full-disk encryption is the v1 requirement**, verified during setup and shown
in the System panel. FileVault on macOS, BitLocker on Windows, LUKS/dm-crypt on
Linux. Where it cannot be determined, Argus reports "unknown" rather than
claiming a pass.

**Application-level encryption of evidence is a tracked follow-up, not
implemented.** With disk encryption on, a stolen machine yields nothing. With it
off, everything is readable — which is why setup checks rather than assumes.

**Retention** deletes evidence after a configurable period (30 days default),
frames, clips and records together. Anything on legal hold or not yet reviewed
is kept. See [docs/DATA_RETENTION.md](docs/DATA_RETENTION.md).

## Threat model

### Defended

| Threat | Control |
|---|---|
| Someone on the machine disables detection | Role enforcement; every change audit-logged with an actor |
| Someone quietly edits the record of what they did | Hash-chained audit log; partial edits detected |
| Live camera feeds read by another process or page | Per-run token on every frame route; localhost bind; no CORS |
| Someone on the site LAN opening the mobile view | Session required on every route; lockout; CSRF on actions; roles enforced |
| Password guessing | Lockout after 5 failures, logged; slow KDF |
| User enumeration via the login | Identical response and work for unknown user and wrong password |
| Stolen or decommissioned machine | Full-disk encryption, verified at setup |
| Evidence kept longer than lawful | Scheduled purge with legal hold |
| Credentials leaking with the evidence database | Separate `auth.db`, mode `0600` |

### Not defended — stated plainly

- **A privileged local attacker.** Anyone with root on the box can read the
  databases and rewrite the entire audit chain. Argus detects partial tampering,
  not a competent full rewrite.
- **A malicious owner.** The Owner role can do everything by design. Their
  actions are logged, but they are not restrained.
- **Network exposure beyond the mobile view.** The console and frame servers
  bind to localhost. The **mobile response view is the one deliberately
  LAN-exposed surface**: every route requires a session against the same
  account store and lockout as the console, the cookie is HttpOnly+SameSite,
  actions carry a CSRF token, roles are enforced (an installer cannot read
  incidents from a phone either), and "no unauthenticated route exists" is a
  named test. Exposing anything else (a reverse proxy to the console, the
  frame publisher) is outside what has been reviewed.
- **The camera feeds themselves.** RTSP credentials are as secure as the
  cameras and network; Argus does not improve them.
- **Model behaviour.** Detection is assistive and measured, not guaranteed.
  Recall and precision with sample sizes and confidence intervals are in
  [docs/NUMBERS.md](docs/NUMBERS.md).
- **Supply chain.** Dependencies are pinned but not vendored or independently
  audited.

## Procurement answers

| Question | Answer |
|---|---|
| Where is data processed? | Entirely on the customer's machine. No cloud, no upload; works fully offline. |
| Is data encrypted at rest? | Via OS full-disk encryption, verified at setup. Application-level encryption is a tracked follow-up. |
| Is data encrypted in transit? | No data leaves the machine. Local endpoints bind to `127.0.0.1`. |
| How are passwords stored? | scrypt, or PBKDF2-HMAC-SHA256 at 600,000 rounds. Never reversibly. |
| Is there role-based access control? | Three roles, enforced server-side, tested by calling past the interface. |
| Is there an audit trail? | Yes — append-only, hash-chained, tamper-evident, Owner-only, separately stored. |
| Can you honour an erasure request? | Yes. Deleting an event removes its frames, clips and record together; an orphan sweep catches anything left behind. |
| What is your data retention? | 30 days by default, per-site configurable, with legal hold. |
| Do you have penetration test results? | No. Not yet commissioned. |
| Do you hold ISO 27001 / SOC 2? | No. |
| Is there a vulnerability disclosure process? | GitHub security advisories on this repository. |
| What happens if a detector fails? | It is counted and surfaced; a component failing more than 1 in 10 attempts is flagged degraded in the System panel. |
| What happens if verification is unavailable? | The alert reaches the operator marked **UNVERIFIED**, never silently dropped. |

## Known gaps

Tracked rather than hidden, in `docs/BACKLOG.md`:

- Application-level encryption of evidence
- Penetration test
- Signed and notarised installers (checksums are published today)
- Audit log shipped off-box for a tamper-proof second copy
