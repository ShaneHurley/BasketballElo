@echo off
REM Double-click to start the T-60 Analysis Dashboard (Windows).
REM Leave this window open while using the site; Ctrl+C to stop.
REM Escape hatches: "Open T-60 Dashboard.bat" --no-browser   or   set T60_NO_BROWSER=1

setlocal enabledelayedexpansion
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

set "NO_BROWSER=0"
if /i "%~1"=="--no-browser" set "NO_BROWSER=1"
if /i "%T60_NO_BROWSER%"=="1" set "NO_BROWSER=1"

REM Ask permission before popping up a browser window.
if "%NO_BROWSER%"=="0" (
  set "ANSWER="
  set /p "ANSWER=Open the T-60 Dashboard in your browser? [Y/n] "
  if /i "!ANSWER!"=="n" set "NO_BROWSER=1"
  if /i "!ANSWER!"=="no" set "NO_BROWSER=1"
)

if "%NO_BROWSER%"=="0" (
  REM Open browser after a short delay (new window so server keeps this console).
  start "" cmd /c "timeout /t 2 /nobreak >nul & start %URL%"
  echo Browser will open at %URL% shortly...
) else (
  echo Browser launch skipped - open %URL% manually when ready.
)

echo T-60 Analysis Dashboard
echo   %URL%
echo   Python: %PY%
echo   Ctrl+C to stop
echo.

%PY% "%CODE%\scripts\run_dashboard.py" --host 127.0.0.1 --port 8765
pause
