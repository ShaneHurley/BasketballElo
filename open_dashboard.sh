#!/bin/bash
# Linux / generic: ./open_dashboard.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/code"
URL="http://127.0.0.1:8765"
cd "$CODE"

if [ -x "$CODE/.venv_dashboard/bin/python" ]; then
  PY="$CODE/.venv_dashboard/bin/python"
else
  PY="python3"
fi

if command -v xdg-open >/dev/null 2>&1; then
  (sleep 1.5 && xdg-open "$URL") &
elif command -v open >/dev/null 2>&1; then
  (sleep 1.5 && open "$URL") &
fi

echo "T-60 Analysis Dashboard → $URL  (Ctrl+C to stop)"
exec "$PY" "$CODE/scripts/run_dashboard.py" --host 127.0.0.1 --port 8765
