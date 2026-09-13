<#
.SYNOPSIS
    Build the Windows desktop package: speech2text-setup.exe

.DESCRIPTION
    Runs PyInstaller over packaging\speech2text.spec, then Inno Setup over
    packaging\installer.iss.

    PyInstaller cannot cross-compile -- it bundles the interpreter and native
    DLLs of the machine it runs on -- so this must run on Windows, with a
    native working directory. A \\wsl.localhost\... path will not work as a
    working directory; copy the tree to something like C:\build\speech2text
    first and run this from there.

.NOTES
    Deliberately installs requirements.txt ONLY. requirements-gpu.txt would add
    ~1.5 GB of CUDA libraries to the bundle and defeat the on-demand download in
    app\cuda_setup.py.
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11",
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }

# -- 1. interpreter --------------------------------------------------------
Step "Checking Python $PythonVersion"
& py "-$PythonVersion" --version
if ($LASTEXITCODE -ne 0) { throw "python $PythonVersion not available via the py launcher" }

$VenvDir = Join-Path $Root ".venv-win"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Step "Creating build virtualenv at $VenvDir"
    & py "-$PythonVersion" -m venv $VenvDir
}

Step "Installing build dependencies"
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt") --quiet
& $VenvPython -m pip install pyinstaller --quiet
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

# -- 2. ffmpeg -------------------------------------------------------------
$BinDir = Join-Path $Root "packaging\bin"
$Ffmpeg = Join-Path $BinDir "ffmpeg.exe"
$Ffprobe = Join-Path $BinDir "ffprobe.exe"
if (-not ((Test-Path $Ffmpeg) -and (Test-Path $Ffprobe))) {
    Step "Fetching a static ffmpeg build"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $Zip = Join-Path $env:TEMP "ffmpeg-release-essentials.zip"
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $Zip
    $Extract = Join-Path $env:TEMP "ffmpeg-extract"
    Remove-Item -Recurse -Force $Extract -ErrorAction SilentlyContinue
    Expand-Archive -Path $Zip -DestinationPath $Extract
    Get-ChildItem -Path $Extract -Recurse -Include "ffmpeg.exe", "ffprobe.exe" |
        ForEach-Object { Copy-Item $_.FullName -Destination $BinDir -Force }
    Remove-Item -Recurse -Force $Extract, $Zip -ErrorAction SilentlyContinue
}
if (-not ((Test-Path $Ffmpeg) -and (Test-Path $Ffprobe))) {
    throw "ffmpeg.exe / ffprobe.exe are missing from $BinDir"
}

# -- 3. bundle -------------------------------------------------------------
Step "Running PyInstaller"
Remove-Item -Recurse -Force (Join-Path $Root "build"), (Join-Path $Root "dist\speech2text") -ErrorAction SilentlyContinue
& $VenvPython -m PyInstaller (Join-Path $Root "packaging\speech2text.spec") `
    --noconfirm --distpath (Join-Path $Root "dist") --workpath (Join-Path $Root "build")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$BundleExe = Join-Path $Root "dist\speech2text\speech2text.exe"
if (-not (Test-Path $BundleExe)) { throw "expected $BundleExe" }
$BundleSize = (Get-ChildItem (Join-Path $Root "dist\speech2text") -Recurse |
    Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("    bundle: {0:N0} MB" -f $BundleSize)

if ($SkipInstaller) { Step "Done (installer skipped)"; exit 0 }

# -- 4. installer ----------------------------------------------------------
Step "Building the installer"
$Iscc = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) { throw "ISCC.exe not found. winget install JRSoftware.InnoSetup" }

& $Iscc (Join-Path $Root "packaging\installer.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

$Setup = Join-Path $Root "dist\speech2text-setup.exe"
$SetupSize = (Get-Item $Setup).Length / 1MB
Write-Host ("`n*** {0}  ({1:N0} MB)" -f $Setup, $SetupSize) -ForegroundColor Green
