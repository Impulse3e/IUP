# Desktop and Start Menu shortcut for IUP Student.
# ASCII-only so Windows PowerShell 5.1 parses it regardless of file encoding.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Exe = Join-Path $Root "dist\IUP Student.exe"
$Bat = Join-Path $Root "scripts\run_student.bat"
$Wsh = New-Object -ComObject WScript.Shell

if (Test-Path $Exe) {
    $Target = $Exe
    $WorkDir = Split-Path $Exe
} else {
    $Target = $Bat
    $WorkDir = $Root
}

function Save-Shortcut([string]$LinkPath) {
    $folder = Split-Path $LinkPath
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
    }
    $lnk = $Wsh.CreateShortcut($LinkPath)
    $lnk.TargetPath = $Target
    $lnk.WorkingDirectory = $WorkDir
    $lnk.WindowStyle = 1
    $lnk.Description = "IUP Student"
    if (Test-Path $Exe) {
        $lnk.IconLocation = "$Exe,0"
    }
    $lnk.Save()
    Write-Host "Shortcut: $LinkPath"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$StartMenu = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"
Save-Shortcut (Join-Path $Desktop "IUP Student.lnk")
Save-Shortcut (Join-Path $StartMenu "IUP Student.lnk")

Write-Host ""
Write-Host "Ready. Double-click IUP Student on the Desktop."
if (-not (Test-Path $Exe)) {
    Write-Host "This shortcut launches scripts\run_student.bat (venv)."
    Write-Host "Standalone exe:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\build_student_windows.ps1"
    Write-Host "Run this script again after the build to point the shortcut at the exe."
}
