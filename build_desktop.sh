#!/usr/bin/env bash
set -euo pipefail

# Build a standalone PubCrawler desktop app (macOS/Linux)
# Requirements: Python 3.8+, pip, pyinstaller, pywebview

APP_NAME="PubCrawlerDesktop"

PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  PYTHON_CMD="python"
fi

echo "[1/4] Checking Python..."
"$PYTHON_CMD" - <<'PY'
import sys
assert sys.version_info >= (3,8), f"Python 3.8+ required, found {sys.version}"
print("Python:", sys.version.split()[0])
PY

echo "[2/4] Ensuring build dependencies (pyinstaller, pywebview) are installed..."
"$PYTHON_CMD" -m pip install --upgrade --quiet pyinstaller pywebview

echo "[3/4] Installing project requirements..."
if [ -f requirements.txt ]; then
  "$PYTHON_CMD" -m pip install --quiet -r requirements.txt
fi

echo "[4/4] Building $APP_NAME..."
rm -rf build dist "$APP_NAME.spec" || true

# Include templates and static assets in the bundle
ADD_DATA=(
  "templates:templates"
  "static:static"
  "favicon.ico:favicon.ico"
)

DATA_ARGS=()
for d in "${ADD_DATA[@]}"; do
  DATA_ARGS+=("--add-data" "$d")
fi

"$PYTHON_CMD" -m PyInstaller \
  --noconfirm \
  --clean \
  --noconsole \
  --name "$APP_NAME" \
  "${DATA_ARGS[@]}" \
  desktop_app.py

echo
echo "✅ Build complete. Find the app in ./dist/$APP_NAME"
echo "   Run: ./dist/$APP_NAME/$APP_NAME"

