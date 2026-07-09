#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/student_launcher_zenity.py"
