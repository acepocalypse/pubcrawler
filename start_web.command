#!/bin/bash
# PubCrawler Web Interface Launcher for macOS
# ===========================================

# Change to the script's directory
cd "$(dirname "$0")"

echo ""
echo "🚀 Starting PubCrawler Web Interface..."
echo ""

# Check for different Python installations
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if it's Python 3
    python_version=$(python --version 2>&1)
    if [[ $python_version == *"Python 3"* ]]; then
        PYTHON_CMD="python"
    else
        echo "❌ Found Python 2, but Python 3 is required"
        echo "Please install Python 3 from https://python.org"
        exit 1
    fi
else
    echo "❌ Python 3 is not installed or not in PATH"
    echo ""
    echo "💡 Troubleshooting:"
    echo "   • Try running 'python3 --version' or 'python --version'"
    echo "   • Install Python 3 from https://python.org"
    echo "   • On macOS: brew install python3"
    exit 1
fi

echo "✅ Found Python: $PYTHON_CMD"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script from the pubcrawler directory"
    echo "The directory should contain app.py"
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Installing requirements..."
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements"
        echo "💡 Try running: $PYTHON_CMD -m pip install --upgrade pip"
        exit 1
    fi
fi

# Generate a random port
port=$((RANDOM % 64511 + 1024))

# Start the web server
echo "✅ Starting web server on http://localhost:$port"
echo ""
echo "💡 Tips:"
echo "   • The browser should open automatically"
echo "   • Use Ctrl+C to stop the server"
echo "   • Check WEB_README.md for more information"
echo ""

$PYTHON_CMD run_web.py --host 127.0.0.1 --port $port
