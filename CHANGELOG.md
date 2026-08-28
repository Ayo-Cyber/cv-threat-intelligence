# Changelog

What changed in each release, written for the person who has to decide whether
to update. Dates are release dates. Every version is built by CI from a tag on
`main` and published with SHA-256 sums — verify your download against them.

## v1.2.0 — 28 Aug 2026

**Windows customers get a real installer.** `argus-windows-setup.exe` now sits
next to the portable zip: Start Menu entry, desktop shortcut, an uninstaller,
and new versions install over old ones instead of beside them. SmartScreen
still warns once — that is the missing signing certificate, not the packaging.

**Live streams no longer fast-forward.** Segmented sources (HLS and similar)
deliver video in multi-second bursts; the wall used to freeze for a segment and
then replay it at decode speed. Bursts now play out smoothly at content rate a
few seconds behind live, catching up quietly when genuinely behind, with lag
hard-capped at eight seconds. Measured on the same public feeds: 0.7-1.2 fps
with five-second lurches before, ~11 fps steady after. Real RTSP cameras
deliver continuously and ride the same path at near-zero lag.

**Several plain-English rules on one camera now all fire.** The scanner used to
ask the model for THE threat — singular — so whichever rule it found most
salient answered every cycle and the rest never fired at all. Every rule is now
checked independently, each with its own cooldown. Also, a threat the model
invents no longer slips through by sharing one generic word with a real rule.

## v1.1.1 — 27 Aug 2026

**Update if you are on v1.1.0.** One user-visible fix, and it is the kind that
looks like a dead product.

- **The live wall no longer goes black after the engine restarts.** Every engine
  restart — switching feeds, a watchdog respawn, or the first start racing the
  first render — issues a new frame-publisher port and token. The wall's tiles
  kept requesting the old one and stayed blank, showing a broken-image icon,
  while clicking a tile to zoom it worked normally. The console's own self-heal
  had already spotted the failure and fetched the new port, but never repainted
  the tiles with it. Tiles that are stale or dead are now repointed within about
  two seconds; healthy tiles are left connected, so one dead camera cannot
  restart every stream on the wall.

Also in this release, affecting measurement rather than the running product:

- **A measurement run now exercises the detector it is named after.** Selecting
  a threat with `--kind` chose which clips to load but not which detectors to
  run, so a "weapons" run could complete, report clean figures with confidence
  intervals, and be describing the shoplifting detector's opinion of the
  footage. The gate was also told every clip was a retail shop, including
  street footage, which points it away from what is being measured. Both are
  fixed and tested. Numbers published before this release were produced under
  the old scene context and are being re-measured.

## v1.1.0 — 27 Aug 2026

**The live wall works.** v1.0.0 shipped with the frame publisher answering
HTTP/1.0, and no browser will progressively render a multipart stream over
HTTP/1.0 — so every tile was blank in that build.

- The wall streams MJPEG instead of polling eleven times a second
- One capture path for every platform; Windows was buffering itself into lag
- Live-edge draining, so a burst from a segmented stream no longer becomes
  permanent delay
- The alerts list ships ~70 KB per tab switch instead of 189 MB
- The 3.3 GB verification model downloads *while* you complete setup, not after
- Several plain-English rules per camera, applied without a restart
- Retention actually deletes expired evidence. It never once did: evidence paths
  were stored relative to the working directory and resolved against the output
  root, so the safety check refused every deletion and the disk simply filled.
  The retention period shown in the app was not being enforced.
- Accounts and the audit log stay global across feeds — per-feed stores had
  fragmented them, so nobody could sign in from a phone on a non-default feed

Installers are smaller across the board: Windows 511 MB, macOS 636 MB,
Linux 859 MB.

## v1.0.0 — 25 Aug 2026

First packaged release: one-click installers for macOS, Windows and Linux, with
the detection engine, the local verification model runtime, and the operator
console in a single download.

---

### Known limitations, all versions

- **The installers are not code-signed.** macOS and Windows will warn that the
  developer is unidentified. Signing certificates are not yet purchased.
- **Segmented live sources (HLS) look jerky.** Argus stays at the live edge, so
  on a source that delivers five seconds of video at a time there is nothing new
  to show between segments. Measured on public YouTube camera feeds: ~1 fps with
  5-second holds, against 11–12 fps on local video and on a continuous stream
  through the identical pipeline. Real RTSP cameras deliver continuously and do
  not behave this way.
- **Seven detectors are built and demonstrable but not yet validated** — panic
  running, person collapsed, weapons, violence, camera tampering, loitering, and
  custom plain-English rules. See `docs/NUMBERS.md`, which states what has been
  measured and what has not.
