# Полная сборка IUP Student.exe (лаунчер + прокторинг, один файл)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if ($env:OS -ne "Windows_NT") {
    Write-Host "Сборка .exe выполняется на Windows с установленным Python."
    exit 1
}

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Сначала запустите: powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1"
    exit 1
}

Write-Host "Скачивание модели MediaPipe (если ещё нет)..."
& $Py scripts\download_mediapipe_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Не удалось скачать модель. Проверьте интернет."
    exit 1
}

Write-Host "Установка PyInstaller..."
& $Py -m pip install --upgrade pip pyinstaller

Write-Host "Сборка (может занять несколько минут)..."
& $Py -m PyInstaller build\windows\iup-student.spec --noconfirm --distpath dist --workpath build\pyinstaller

$Exe = Join-Path $Root "dist\IUP Student.exe"
if (Test-Path $Exe) {
    $SizeMb = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Готово: $Exe ($SizeMb MB)"
    Write-Host ""
    Write-Host "Скопируйте IUP Student.exe на компьютер студента."
    Write-Host "Дополнительная установка не требуется — вход, запись на экзамен и прокторинг в одном файле."
} else {
    Write-Host "Ошибка сборки — проверьте лог PyInstaller в build\pyinstaller"
    exit 1
}
