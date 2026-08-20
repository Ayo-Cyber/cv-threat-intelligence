# Code signing & notarisation (EP-05-T2)

For a **security product**, asking a customer to click past an OS malware
warning at install time undermines the exact trust the product sells. This
page is the complete recipe: what to buy, what to set, and what CI then does
automatically. It is also the bus-factor document the audit asked for — if
the laptop dies, everything below is recoverable from accounts + this file.

## What to buy (one-time, owner action)

| Platform | Account | Produces | Cost |
|---|---|---|---|
| macOS | [Apple Developer Program](https://developer.apple.com/programs/) (the company, not a personal ID) | **Developer ID Application** certificate + notarisation access | $99/yr |
| Windows | An Authenticode certificate from a CA (SSL.com, Certum, DigiCert…). **OV** works but builds SmartScreen reputation slowly; **EV** skips straight past SmartScreen | code-signing cert (`.pfx` or cloud HSM) | ~$70–400/yr |

## Key handling (bus factor)

- **macOS**: export the Developer ID Application cert + private key from
  Keychain as a password-protected `.p12`. Keep it in the company password
  manager, NOT in the repo, NOT only on one laptop. If lost: revoke and
  re-issue from the Apple Developer portal — nothing else breaks.
- **Apple notarisation**: create an [app-specific password](https://appleid.apple.com)
  for the Apple ID; store alongside the p12.
- **Windows**: modern CAs issue on a cloud HSM or USB token; if a `.pfx` is
  possible, store it exactly like the p12.
- The **update-channel signing key** (`~/.argus-release.key`, Ed25519, used by
  `tools/make_update.py`) is separate from all of the above and already
  documented there. OS signing proves the installer's origin to the OS;
  the update key proves updates' origin to Argus itself.

## GitHub secrets CI expects

Set under *Settings → Secrets and variables → Actions*:

| Secret | Content |
|---|---|
| `MACOS_CERT_P12` | the `.p12`, base64: `base64 -i cert.p12 \| pbcopy` |
| `MACOS_CERT_PASSWORD` | its export password |
| `MACOS_SIGN_IDENTITY` | e.g. `Developer ID Application: Argus Ltd (TEAMID123)` |
| `APPLE_ID` | the notarising Apple ID email |
| `APPLE_TEAM_ID` | 10-char team id |
| `APPLE_APP_PASSWORD` | the app-specific password |
| `WINDOWS_CERT_PFX` | the `.pfx`, base64 (if file-based signing) |
| `WINDOWS_CERT_PASSWORD` | its password |

**Until the secrets exist, CI builds exactly what it builds today — unsigned,
and says so in the step log.** No secret is ever required for the build to
succeed; signing engages by presence.

## What CI does once the secrets exist

macOS (in `build-app.yml`):
1. import the p12 into a throwaway keychain
2. `codesign --deep --force --options runtime --entitlements packaging/entitlements.plist` over `dist/Argus.app` (hardened runtime — required by notarisation)
3. rebuild the dmg from the signed app, sign the dmg
4. `xcrun notarytool submit dist/Argus.dmg --wait` then `xcrun stapler staple`

Windows:
1. decode the pfx
2. `signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256` over `dist/Argus/Argus.exe` and `argus-engine.exe`

## Verifying (the acceptance test)

On a machine that has **never seen the app**:
- macOS: download the dmg from the release page, open it, drag, launch.
  There must be **no** "unidentified developer" / "Apple could not verify"
  dialog. `spctl -a -vv /Applications/Argus.app` must say `accepted · notarized`.
- Windows: run the installer; SmartScreen must not interpose (EV) or must
  show the publisher name, not "Unknown publisher" (OV, after reputation).
