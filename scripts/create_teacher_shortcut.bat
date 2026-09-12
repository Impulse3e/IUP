@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_teacher_shortcut.ps1"
if errorlevel 1 (
    echo Shortcut script failed.
    pause
    exit /b 1
)
pause
