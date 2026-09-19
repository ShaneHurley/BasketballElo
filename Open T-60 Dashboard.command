#!/bin/bash
# Double-click in Finder (macOS) to start the T-60 Analysis Dashboard.
# Keep this Terminal window open while using the site; Ctrl+C to stop.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/code"
URL="http://127.0.0.1:8765"

cd "$CODE"

if [ -x "$CODE/.venv_dashboard/bin/python" ]; then
  PY="$CODE/.venv_dashboard/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "Python 3 not found. Install Python or create .venv_dashboard first."
  echo "  cd code && python3 -m venv .venv_dashboard && .venv_dashboard/bin/pip install -r dashboard/requirements.txt"
  read -r -p "Press Enter to close…"
  exit 1
fi

# Open the browser shortly after the server starts.
(sleep 1.5 && open "$URL") &

echo "T-60 Analysis Dashboard"
echo "  $URL"
echo "  Python: $PY"
echo "  Ctrl+C to stop"
echo

exec "$PY" "$CODE/scripts/run_dashboard.py" --host 127.0.0.1 --port 8765
