$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONPATH = "."
$env:PYTHONUTF8 = "1"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Сначала установите окружение: scripts\setup_windows.ps1"
    exit 1
}
& $Py -m server.app.backup
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Копия сохранена в data\backups\"
