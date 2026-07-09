# Установка IUP Student на Windows (PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Find-Python {
    $candidates = @("python", "py", "python3")
    foreach ($name in $candidates) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    return $null
}

$Python = Find-Python
if (-not $Python) {
    Write-Host "Python не найден. Установите Python 3.11+ с https://www.python.org/downloads/"
    Write-Host "При установке отметьте «Add python.exe to PATH» и «tcl/tk»."
    exit 1
}

Write-Host "Python: $Python"

if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}

$Pip = Join-Path $Root ".venv\Scripts\pip.exe"
$Py = Join-Path $Root ".venv\Scripts\python.exe"

& $Pip install --upgrade pip
& $Pip install httpx
& $Pip install -r agent\requirements.txt

Write-Host ""
Write-Host "Готово. Запуск приложения участника:"
Write-Host "  scripts\run_student.bat"
Write-Host ""
Write-Host "Или соберите отдельный .exe:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\build_student_windows.ps1"
