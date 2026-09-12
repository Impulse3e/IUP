#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

if .venv/bin/python -c "import tkinter" 2>/dev/null; then
  exec .venv/bin/python -m agent.launcher
fi

if command -v zenity >/dev/null 2>&1; then
  exec ./scripts/run_student_zenity.sh
fi

URL="${IUP_SERVER_URL:-http://localhost:8000}/student"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL"
  zenity --info --text="Открыт кабинет участника в браузере.\n\nДля GUI-приложения установите:\n  sudo pacman -S tk zenity" 2>/dev/null || \
    echo "Открыт $URL — установите tk или zenity для приложения"
  exit 0
fi

echo "Установите GUI-зависимости:"
echo "  sudo pacman -S tk zenity"
echo "Или откройте в браузере: $URL"
exit 1
