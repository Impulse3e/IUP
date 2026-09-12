# Установка IUP Student на Windows (PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Find-Python {
    $named = @("python", "py", "python3")
    foreach ($name in $named) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notmatch "WindowsApps") {
            return $cmd.Source
        }
    }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
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

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Создан .env из .env.example"
}

Write-Host ""
Write-Host "Готово. Запуск приложения участника:"
Write-Host "  scripts\run_student.bat"
Write-Host "  вход: student@iup.local / student123"
Write-Host ""
Write-Host "Ярлык на рабочем столе:"
& (Join-Path $Root "scripts\create_student_shortcut.ps1")
Write-Host ""
Write-Host "Сервер (если ещё не поднят):"
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1'
Write-Host "  scripts\run_server.bat"
Write-Host ""
Write-Host "Или соберите отдельный .exe:"
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\build_student_windows.ps1'
