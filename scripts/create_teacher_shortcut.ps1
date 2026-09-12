# Desktop and Start Menu shortcut for the teacher dashboard + server.
# ASCII-only so Windows PowerShell 5.1 parses it regardless of file encoding.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Bat = Join-Path $Root "scripts\run_teacher.bat"
$Wsh = New-Object -ComObject WScript.Shell

function Save-Shortcut([string]$LinkPath) {
    $folder = Split-Path $LinkPath
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
    }
    $lnk = $Wsh.CreateShortcut($LinkPath)
    $lnk.TargetPath = $Bat
    $lnk.WorkingDirectory = $Root
    $lnk.WindowStyle = 1
    $lnk.Description = "IUP Teacher - start server and open dashboard"
    $lnk.Save()
    Write-Host "Shortcut: $LinkPath"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$StartMenu = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs"
Save-Shortcut (Join-Path $Desktop "IUP Teacher.lnk")
Save-Shortcut (Join-Path $StartMenu "IUP Teacher.lnk")

Write-Host ""
Write-Host "Ready. Double-click IUP Teacher on the Desktop."
Write-Host "It starts the server and opens http://127.0.0.1:8000"
Write-Host "Login: admin@iup.local / admin123"
