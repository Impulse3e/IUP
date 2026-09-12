@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "PYTHONPATH=."
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
    echo Не найдено виртуальное окружение .venv
    echo Запустите: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
    pause
    exit /b 1
)

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    echo Создан .env из .env.example
)

echo IUP server: http://localhost:8000
echo Student:    http://localhost:8000/student
".venv\Scripts\python.exe" -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir server
