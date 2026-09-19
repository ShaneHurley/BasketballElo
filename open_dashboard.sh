#!/bin/bash
# Linux / generic: ./open_dashboard.sh
# Escape hatches: ./open_dashboard.sh --no-browser   or   T60_NO_BROWSER=1 ./open_dashboard.sh
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

NO_BROWSER=0
for arg in "$@"; do
  case "$arg" in
    --no-browser) NO_BROWSER=1 ;;
  esac
done
if [ "${T60_NO_BROWSER:-0}" = "1" ]; then
  NO_BROWSER=1
fi

OPEN_CMD=""
if command -v xdg-open >/dev/null 2>&1; then
  OPEN_CMD="xdg-open"
elif command -v open >/dev/null 2>&1; then
  OPEN_CMD="open"
fi

# Ask permission before popping up a browser window.
if [ "$NO_BROWSER" -eq 0 ] && [ -n "$OPEN_CMD" ]; then
  if [ -t 0 ]; then
    printf "Open the T-60 Dashboard in your browser? [Y/n] "
    read -r answer
    case "$answer" in
      [nN]|[nN][oO]) NO_BROWSER=1 ;;
    esac
  else
    # No interactive terminal (e.g. launched from a script) — don't pop windows.
    NO_BROWSER=1
  fi
fi

if [ "$NO_BROWSER" -eq 0 ] && [ -n "$OPEN_CMD" ]; then
  (sleep 1.5 && "$OPEN_CMD" "$URL") &
  echo "Browser will open at $URL shortly…"
else
  echo "Browser launch skipped — open $URL manually when ready."
fi

echo "T-60 Analysis Dashboard → $URL  (Ctrl+C to stop)"
exec "$PY" "$CODE/scripts/run_dashboard.py" --host 127.0.0.1 --port 8765
