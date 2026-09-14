# Build IUP Student.exe (launcher + proctoring, one file)
# ASCII-only so Windows PowerShell 5.1 can parse it.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if ($env:OS -ne "Windows_NT") {
    Write-Host "Build the .exe on Windows with Python installed."
    exit 1
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Install the student venv first:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1"
    exit 1
}

Write-Host "Downloading MediaPipe model if needed..."
& $Py scripts\download_mediapipe_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not download the model. Check the internet connection."
    exit 1
}

Write-Host "Installing PyInstaller..."
& $Py -m pip install --upgrade pip pyinstaller
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building (this can take several minutes)..."
& $Py -m PyInstaller build\windows\iup-student.spec --noconfirm --distpath dist --workpath build\pyinstaller
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Exe = Join-Path $Root "dist\IUP Student.exe"
if (Test-Path $Exe) {
    $SizeMb = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Ready: $Exe ($SizeMb MB)"
    Write-Host "Copy this file to the student PC. No extra install is required."
    Write-Host ""
    Write-Host "Updating desktop shortcut..."
    & (Join-Path $Root "scripts\create_student_shortcut.ps1")
} else {
    Write-Host "Build failed. Check the PyInstaller log in build\pyinstaller"
    exit 1
}
