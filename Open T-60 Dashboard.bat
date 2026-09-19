@echo off
REM Double-click to start the T-60 Analysis Dashboard (Windows).
REM Leave this window open while using the site; Ctrl+C to stop.

setlocal
set "ROOT=%~dp0"
set "CODE=%ROOT%code"
set "URL=http://127.0.0.1:8765"

cd /d "%CODE%" || (
  echo Could not find code\ folder next to this script.
  pause
  exit /b 1
)

set "PY="
if exist "%CODE%\.venv_dashboard\Scripts\python.exe" (
  set "PY=%CODE%\.venv_dashboard\Scripts\python.exe"
) else (
  where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3 not found. Install Python or create .venv_dashboard first.
  echo   cd code ^&^& python -m venv .venv_dashboard ^&^& .venv_dashboard\Scripts\pip install -r dashboard\requirements.txt
  pause
  exit /b 1
)

REM Open browser after a short delay (new window so server keeps this console).
start "" cmd /c "timeout /t 2 /nobreak >nul & start %URL%"

echo T-60 Analysis Dashboard
echo   %URL%
echo   Python: %PY%
echo   Ctrl+C to stop
echo.

%PY% "%CODE%\scripts\run_dashboard.py" --host 127.0.0.1 --port 8765
pause
