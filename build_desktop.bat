@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Build a standalone PubCrawler desktop app (Windows)
REM Requirements: Python 3.8+, pip, pyinstaller, pywebview

set APP_NAME=PubCrawlerDesktop

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Python not found in PATH.
  exit /b 1
)

echo [1/4] Checking Python...
python - <<PY
import sys
assert sys.version_info >= (3,8), f"Python 3.8+ required, found {sys.version}"
print("Python:", sys.version.split()[0])
PY
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [2/4] Ensuring build dependencies (pyinstaller, pywebview)...
python -m pip install --upgrade --quiet pyinstaller pywebview
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo [3/4] Installing project requirements...
if exist requirements.txt (
  python -m pip install --quiet -r requirements.txt
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

echo [4/4] Building %APP_NAME%...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist %APP_NAME%.spec del /q %APP_NAME%.spec

REM Include templates and static assets in the bundle
REM On Windows, use ';' to separate src;dest
set DATA_ARGS=--add-data "templates;templates" --add-data "static;static" --add-data "favicon.ico;favicon.ico"

python -m PyInstaller --noconfirm --clean --noconsole --name "%APP_NAME%" %DATA_ARGS% desktop_app.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo.
echo ✅ Build complete. Find the app in .\dist\%APP_NAME%
echo    Run: .\dist\%APP_NAME%\%APP_NAME%.exe

endlocal

