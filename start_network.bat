@echo off
REM PubCrawler Web Interface Network Launcher for Windows
REM ======================================================

echo.
echo 🚀 Starting PubCrawler Web Interface for Network Access...
echo.

REM Check for different Python installations (including Anaconda/Conda)
set PYTHON_CMD=
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)

REM Check for Anaconda/Conda installations
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\anaconda3\python.exe"
    goto :python_found
)

if exist "C:\Users\anaconda3\python.exe" (
    set PYTHON_CMD="C:\Users\anaconda3\python.exe"
    goto :python_found
)

if exist "%LOCALAPPDATA%\anaconda3\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\anaconda3\python.exe"
    goto :python_found
)

REM Check for Miniconda
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set PYTHON_CMD="%USERPROFILE%\miniconda3\python.exe"
    goto :python_found
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_found
)

echo ❌ Python is not installed or not in PATH
echo.
echo 💡 Quick Fix:
echo    Run 'fix_python_path.bat' to detect and fix Python PATH issues
echo.
echo 💡 Manual Troubleshooting:
echo    • Try running 'py --version' in Command Prompt
echo    • If that works, Python is installed but PATH needs fixing
echo    • Download Python from https://python.org if not installed
echo    • Make sure to check "Add Python to PATH" during installation
echo.
echo 🔧 Alternative: Try running fix_python_path.bat first
echo.
pause
exit /b 1

:python_found
echo ✅ Found Python: %PYTHON_CMD%

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
    %PYTHON_CMD% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        echo 💡 Try running: %PYTHON_CMD% -m pip install --upgrade pip
        pause
        exit /b 1
    )
)

REM Start the web server
echo ✅ Your network URL is: http://10.165.41.221:52123
echo.
echo 💡 Tips:
echo    • Share the network URL above with your coworkers
echo    • Use Ctrl+C to stop the server
echo    • Check WEB_README.md for more information
echo.

%PYTHON_CMD% run_web.py --host 0.0.0.0 --port 52123

pause
