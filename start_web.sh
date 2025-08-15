#!/bin/bash
# PubCrawler Web Interface Launcher for Linux/Mac
# ===============================================

echo ""
echo "🚀 Starting PubCrawler Web Interface..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script from the pubcrawler directory"
    echo "The directory should contain app.py"
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "📦 Installing requirements..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements"
        exit 1
    fi
fi

# Start the web server
echo "✅ Starting web server on http://localhost:5000"
echo ""
echo "💡 Tips:"
echo "   • The browser should open automatically"
echo "   • Use Ctrl+C to stop the server"
echo "   • Check WEB_README.md for more information"
echo ""

python3 run_web.py --host 127.0.0.1 --port 5000
