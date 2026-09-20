# Changelog

What changed in each release, written for the person who has to decide whether
to update. Dates are release dates. Every version is built by CI from a tag on
`main` and published with SHA-256 sums — verify your download against them.

## v1.8.15 — 20 Sep 2026

**Your cameras show a picture again.** Adding a camera and pressing Start
Monitoring stopped the detection engine dead, on every camera, not only the
new one. The camera would test as reachable, the button would quietly flip
back, and no video ever arrived. The cause: a camera added through the app
carries no detection settings until you pick what the site is for, and the
engine treated that as a fatal error rather than a choice you had not made
yet. A camera with no use case selected now runs the general detection set,
and choosing a use case still replaces it. **Upgrade if you added a camera on
v1.8.14 and saw no picture — nothing was wrong with your camera.**

**When monitoring stops, Argus now tells you why.** The app reported a
crashed engine exactly the same way it reported one you had stopped on
purpose: stopped, no explanation. It now shows the reason and where the log
is, so a problem arrives as a message on screen instead of a black rectangle.

**Adding your first camera no longer asks where it goes before anywhere
exists.** On a new site the Branch and Area fields were required but empty,
because branches are created elsewhere. The first camera now starts unplaced,
and the app says where to place it once you have set your locations up.

## v1.8.14 — 18 Sep 2026

**Argus now opens on your site, not on a sample one.** Every launch started in
a demonstration workspace: a fixture site with invented cameras and invented
incidents, already signed in as nobody. A first-time user never saw the
sign-up screen, never saw sign-in, and never saw site setup — the only way to
the real thing was a small Demo / Local engine switch in the top bar that
nothing drew attention to. The desktop app now opens on your own installation,
which on a new machine means creating your owner account. The demonstration
workspace is still there, one click away, and it stays selected if you choose
it.

**Setting up a new site walks you through it.** Sign in to a site that has not
been set up yet and Argus opens the setup wizard — cameras, scenes and zones,
what the site is for, which detectors to run, how verification behaves, and a
final check — instead of an empty dashboard with no obvious first move.

**Connecting a camera no longer requires knowing RTSP.** Adding a camera asked
for a stream address, which is fine if you already know that a Hikvision
substream lives at /Streaming/Channels/102 and a Tapo at /stream2, and useless
if you don't. Enter the camera's IP address, pick the make, and type the
username and password: Argus builds the address. The password is masked while
you type it, with a reveal button, and passwords containing symbols such as @
or : are handled correctly instead of corrupting the address. You can still
type a full address by hand.

**Camera discovery on the Add Camera screen works again.** Scanning the network
for cameras had been returning "no such camera" — the request was being
answered by the wrong handler. Anyone who tried it got an error with no
explanation.

**Everything the engine can do is reachable from the interface again.** Moving
to the new interface left a long tail of capabilities with no way to invoke
them: PDF incident reports, evidence export, legal holds, weekly summaries,
shift handovers, backups, the audit trail, account administration, model
downloads and per-camera rules. All of it is available from the System screen.
Several of these deserve screens of their own and will get them; being
reachable comes first.

## v1.8.13 — 17 Sep 2026

**The new interface is finally the one you install.** Argus has had a rebuilt
desktop UI since 9 September, and every release since shipped the old one
anyway: the frontend was merged as source but no build step ever compiled or
packaged it, and nothing in CI checked. Five releases went out carrying an
interface nobody had been demonstrating. The installer now *is* that new UI,
with the detection engine packaged inside it.

**What changes when you upgrade.** Your site configuration, accounts and
recorded events stay where they are and open as before — the new shell reads
the same per-user data directory the old one wrote to. The Windows installer
keeps its name and installs over the top. The old interface is still inside
the installation as a support fallback; nothing launches it.

**Account recovery now works on an installed machine.** Recovery had only ever
existed as a script for someone with the source code and a Python environment,
which no customer has — a locked-out installation had no way back in. The
shipped engine exposes the same recovery flow, with the same protections, and
the app shows the exact command to run.

**A build can no longer ship without its interface.** The release now fails if
the packaged app is missing the renderer, either engine binary, or their model
weights — the check whose absence caused all of this.

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
