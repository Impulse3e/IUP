#!/usr/bin/env bash
# Установка зависимостей без активации venv (работает в bash и fish)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

PIP=.venv/bin/pip
PY=.venv/bin/python

"$PIP" install --upgrade pip
"$PIP" install "pydantic>=2.11" "pydantic-settings>=2.7"
"$PIP" install -r server/requirements.txt || true
"$PIP" install -r agent/requirements.txt

echo ""
echo "Готово. Запуск без source activate:"
echo "  ./scripts/run_server.sh"
echo "  ./scripts/run_agent.sh --token <TOKEN>"
