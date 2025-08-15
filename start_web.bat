@echo off
REM PubCrawler Web Interface Launcher for Windows
REM =============================================

echo.
echo 🚀 Starting PubCrawler Web Interface...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ Please run this script from the pubcrawler directory
    echo The directory should contain app.py
    pause
    exit /b 1
)

REM Install requirements if needed
if not exist "venv" (
    echo 📦 Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
)

REM Start the web server
echo ✅ Starting web server on http://localhost:5000
echo.
echo 💡 Tips:
echo    • The browser will open automatically
echo    • Use Ctrl+C to stop the server
echo    • Check WEB_README.md for more information
echo.

python run_web.py --host 127.0.0.1 --port 5000

pause
