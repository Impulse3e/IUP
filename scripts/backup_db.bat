@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
set "PYTHONPATH=."
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo Не найдено виртуальное окружение .venv
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m server.app.backup
if errorlevel 1 (
    echo Не удалось создать копию базы.
    pause
    exit /b 1
)
echo Копия сохранена в data\backups\
pause
