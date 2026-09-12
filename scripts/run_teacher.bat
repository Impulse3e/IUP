@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "PYTHONPATH=."
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv
    echo Run: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
    pause
    exit /b 1
)

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
)

".venv\Scripts\python.exe" -c "import socket; s=socket.socket(); s.settimeout(0.4); r=s.connect_ex(('127.0.0.1',8000)); s.close(); raise SystemExit(0 if r==0 else 1)"
if %errorlevel%==0 (
    echo IUP server already running.
    start "" "http://127.0.0.1:8000/"
    exit /b 0
)

echo Starting IUP server...
echo Teacher:  http://127.0.0.1:8000
echo Student:  http://127.0.0.1:8000/student
echo Login:    admin@iup.local / admin123
echo.
echo Keep this window open while the exam is running.
echo Close it to stop the server.
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000/"
".venv\Scripts\python.exe" -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000
