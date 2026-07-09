#!/usr/bin/env fish
# Установка для fish shell
cd (dirname (status filename))/..
if not test -d .venv
    python3 -m venv .venv
end
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "pydantic>=2.11" "pydantic-settings>=2.7"
.venv/bin/pip install -r server/requirements.txt; or true
.venv/bin/pip install -r agent/requirements.txt
echo "Готово:"
echo "  ./scripts/run_server.sh"
echo "  ./scripts/run_agent.sh --token <TOKEN>"
