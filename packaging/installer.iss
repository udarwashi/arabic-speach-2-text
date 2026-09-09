; Inno Setup script for the Arabic speech-to-text desktop build.
;
; Per-user by design: installing under %LOCALAPPDATA%\Programs needs no
; administrator rights, so the user never sees a UAC prompt, and it works on a
; locked-down work laptop.
;
; Build with:  ISCC.exe packaging\installer.iss

#define AppName "Arabic Speech to Text"
#define AppNameAr "تحويل الصوت إلى نص"
#define AppVersion "1.0.0"
#define AppExe "speech2text.exe"

[Setup]
AppId={{7F3B1C42-9A5E-4D18-B6C0-2E8A5D91F4A7}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\speech2text
DefaultGroupName={#AppNameAr}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=speech2text-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#AppNameAr}
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut / إنشاء اختصار على سطح المكتب"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\speech2text\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppNameAr}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppNameAr}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start now / تشغيل البرنامج الآن"; Flags: nowait postinstall skipifsilent

[Code]
// The model weights and the CUDA runtime live outside the program directory,
// in %LOCALAPPDATA%\speech2text, and are several gigabytes. Re-downloading them
// after an accidental uninstall is worse than leaving a cache behind, so they
// are kept unless the user explicitly asks otherwise.
//
// Note: Pascal block comments end at the first closing brace, so a brace-quoted
// comment must never mention an Inno constant such as the app directory one.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\speech2text');
    if DirExists(DataDir) then
    begin
      if MsgBox('Also delete the downloaded models and GPU libraries (several GB)?'
                + #13#10 + 'حذف النماذج ومكتبات التسريع المخزّنة أيضاً؟'
                + #13#10 + #13#10
                + 'Choose No to keep them if you plan to reinstall.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
