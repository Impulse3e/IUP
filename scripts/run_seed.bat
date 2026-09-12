@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "PYTHONPATH=."

if not exist ".venv\Scripts\python.exe" (
    echo Не найдено виртуальное окружение .venv
    echo Запустите: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\seed_demo.py
