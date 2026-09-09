# Zone Editing and Account Access

Approved scope: full-screen proportional zone drawing, local-machine password
recovery, and owner-authorized account creation. Keep detector logic and existing
camera configurations unchanged. Continue on the current UI worktree.

- [x] Reproduce zone viewport clipping with landscape/portrait frames, desktop and mobile. Expand the existing camera dialog for Zones; fit both dimensions, preserve original coordinates and protect unsaved drawings.
- [x] Add a terminal-only recovery script using AccountStore and AuditLog. Require explicit selection/confirmation, hidden password entry, retain all users/evidence, revoke target sessions and clear target lockout. Do not expose account override through IPC.
- [x] Add login recovery/create-account guidance and owner-only Users settings with existing add_user authorization.
- [x] Run unit, Python security, browser geometry and native Electron account/zone tests. Rebuild source-run app and document exact recovery usage.

Results: 12 Vitest tests, 12 desktop Python tests, 9 browser workflow tests,
48 existing backend security/recovery tests, and the native Electron account/zone
smoke check passed. Production assets rebuilt. Inspected native recovery and
portrait/landscape zone screenshots. No real user passwords reset, no detector
inference started by these checks, and no commit/push performed. Windows remains
untested natively. Keep existing local camera/rule edits untouched.

Verification: backend polygon fixture, bottom-edge rectangle save, no image
distortion at 1440x900 / 1024x768 / 390x844, Escape discard guard; recovery refuses
unknown users/short passwords, preserves other accounts and evidence, audit chain
intact; signed-out/operator cannot add users; owner creates account and new user
can sign in. No automatic commit or push.
