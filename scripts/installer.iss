; Inno Setup script — the Argus Windows installer.
;
; Until 28 Aug this file described an app that no longer exists (CVTI.exe,
; v0.1.0) and no build ever ran it — Windows customers got a bare zip while
; macOS got a dmg: unzip somewhere, find Argus.exe in a folder of DLLs, no
; Start Menu entry, no uninstaller. CI now compiles this on every release
; (Inno Setup is preinstalled on GitHub's windows runners) and ships
; argus-windows-setup.exe alongside the portable zip.
;
;   ISCC.exe /DAppVersion=1.2.3 scripts\installer.iss
;
; Being unsigned, the setup exe still triggers SmartScreen — an Authenticode
; certificate fixes that (docs/SIGNING.md), not this script. What this script
; fixes is everything after the warning.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "Argus"
#define AppExeName "Argus.exe"
#define AppPublisher "Argus"
#define BundleDir "..\dist\Argus"

[Setup]
; Stable AppId so upgrades install over the top instead of side-by-side.
AppId={{7E3F2A91-4C58-4B7D-9A16-ARGUS0000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
; 64-bit app, 64-bit install: without these Windows filed a 1.4 GB
; PyInstaller bundle under 'Program Files (x86)' (pilot log, 1 Sep).
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=argus-windows-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequiredOverridesAllowed=commandline dialog
UninstallDisplayIcon={app}\{#AppExeName}
; The bundle is ~1.4 GB unpacked; make the disk math visible up front.
ExtraDiskSpaceRequired=524288000

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";          Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}";  Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
