@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo IUP: inbound TCP 8000 (LAN)
echo Need Administrator. Confirm the UAC prompt if it appears.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0allow_firewall_8000.ps1"
if errorlevel 1 (
    echo Failed to add the firewall rule.
    pause
    exit /b 1
)
pause
