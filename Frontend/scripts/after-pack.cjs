/**
 * Ad-hoc sign the macOS app when no Apple certificate is available.
 *
 * Without this, electron-builder is told CSC_IDENTITY_AUTO_DISCOVERY=false and
 * ships the .app with NO signature at all. On Apple Silicon every executable
 * must carry at least an ad-hoc signature, so macOS does not say "unsigned" —
 * it says **"Argus is damaged and can't be opened. You should move it to the
 * Trash."** (reported 24 Sep on a fresh v1.8.22 download). That wording makes
 * a working build look corrupt, and the only way past it is a Terminal
 * command most operators will never run.
 *
 * An ad-hoc signature (`codesign --sign -`) costs nothing, needs no Apple
 * account, and downgrades that hard refusal to the ordinary "unidentified
 * developer" dialog, which right-click -> Open clears. The real fix is a
 * Developer ID plus notarisation (docs/SIGNING.md); this is what we can do
 * until those secrets exist, and it is a strict improvement over shipping
 * something macOS calls damaged.
 */
const { execFileSync } = require("node:child_process");
const path = require("node:path");

/** Sign only a Mac build that nothing else is going to sign. */
function shouldAdHocSign(platformName, env = process.env) {
  if (platformName !== "darwin") return false;
  // A real certificate is present: electron-builder signs properly, leave it be.
  return !env.CSC_LINK && !env.MACOS_CERT_P12;
}

exports.shouldAdHocSign = shouldAdHocSign;

exports.default = async function afterPack(context) {
  if (!shouldAdHocSign(context.electronPlatformName)) return;
  const app = path.join(
    context.appOutDir,
    `${context.packager.appInfo.productFilename}.app`,
  );
  console.log(`ad-hoc signing ${app} (no Apple certificate — see docs/SIGNING.md)`);
  // --deep so the bundled engine's binaries are covered too; --force replaces
  // PyInstaller's own ad-hoc signatures, which is what we want for one
  // consistent seal over the tree macOS is about to judge.
  execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], {
    stdio: "inherit",
  });
  execFileSync("codesign", ["--verify", "--deep", "--strict", app], {
    stdio: "inherit",
  });
  console.log("ad-hoc signature verified");
};
