# Signed updates

*How a fix reaches a site without a site visit — and why a site will not
install anything else.*

## The trust model

- Every update is **Ed25519-signed** with the vendor release key. The public
  half is embedded in the app (`cvti/updates.py`); the private half stays with
  the vendor, offline. A site installs nothing that key did not sign — there is
  no "skip verification" option, and a missing signature is treated exactly
  like a forged one.
- The **signature covers the archive bytes**, and the archive is re-hashed
  locally before unpacking — a manifest that lies about its hash changes
  nothing.
- **The customer controls timing.** Checking for an update reads a manifest and
  nothing else. Installing happens only when a person asks. Nothing
  auto-updates mid-shift, ever.
- **Rollback needs no network**: the previous version's files stay on disk, and
  switching back is a pointer change.

## Site side

```
<install>/versions/<version>/...   unpacked releases (two kept: current + rollback)
<install>/current.json             {"version": ..., "previous": ...}
```

The launcher resolves `current.json` first. If a new version fails to start,
`rollback()` restores the previous one — one call, offline.

## Vendor side

```bash
# once, ever — guard this file with your life
python tools/make_update.py keygen --out release.key

# per release
python tools/make_update.py sign --key release.key --version 0.10.0 \
    --dir dist/CVTI-Console --url https://your-host/argus-0.10.0.zip \
    --notes "what changed" --out-dir updates/
```

`sign` zips the build, signs it, writes `manifest.json`, and **verifies the
result against the public key embedded in the app** — so signing with the
wrong key is caught at build time, not by every site refusing.

Host the zip and manifest anywhere static (a GitHub release works). Sites point
their update URL at the manifest.

## If the key is lost or leaked

Lost: no further updates can be shipped; sites keep running what they have.
Leaked: whoever holds it can sign updates sites will trust — rotate by shipping
one final update, signed with the old key, that carries the new public key.
Either way this is the product's root of trust: keep it off the repository, off
the sites, and backed up somewhere that is not this laptop.
