@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "PYTHONPATH=."
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if exist "dist\IUP Student.exe" (
    start "" "dist\IUP Student.exe"
    exit /b 0
)

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m agent.launcher
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m agent.launcher
    exit /b 0
)

echo Не найдено ни dist\IUP Student.exe, ни виртуальное окружение .venv
echo Запустите установку: powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
pause
exit /b 1
