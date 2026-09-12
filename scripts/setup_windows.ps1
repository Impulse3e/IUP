# Локальная подготовка IUP на Windows: Python venv, сервер, .env, демо-данные.
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
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python313\python.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            return $path
        }
    }
    $versions = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python" -ErrorAction SilentlyContinue
    foreach ($dir in $versions) {
        $exe = Join-Path $dir.FullName "python.exe"
        if (Test-Path $exe) {
            return $exe
        }
    }
    return $null
}

$Python = Find-Python
if (-not $Python) {
    Write-Host "Python не найден. Установите Python 3.11+ с https://www.python.org/downloads/"
    Write-Host "При установке отметьте «Add python.exe to PATH»."
    exit 1
}

Write-Host "Python: $Python"
& $Python --version

if (-not (Test-Path ".venv")) {
    Write-Host "Создаю .venv..."
    & $Python -m venv .venv
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "Устанавливаю зависимости сервера..."
& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Py -m pip install -r server\requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Создан .env из .env.example"
}

New-Item -ItemType Directory -Force -Path "data", "data\storage" | Out-Null
$env:PYTHONPATH = $Root
$env:PYTHONIOENCODING = "utf-8"
Write-Host "Сидирую демо-данные..."
& $Py scripts\seed_demo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Ярлык преподавателя (сервер + панель):"
& (Join-Path $Root "scripts\create_teacher_shortcut.ps1")
Write-Host ""
Write-Host "Готово. Дальше:"
Write-Host "  1. Ярлык:    IUP Teacher на рабочем столе"
Write-Host "  2. Или:      scripts\run_teacher.bat"
Write-Host "  3. Панель:   http://localhost:8000     (admin@iup.local / admin123)"
Write-Host "  4. Студент:  http://localhost:8000/student  (student@iup.local / student123)"
Write-Host ""
Write-Host "Клиент участника (камера/микрофон) ставится отдельно:"
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1'
