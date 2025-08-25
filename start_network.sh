#!/bin/bash
# PubCrawler Web Interface Network Launcher for macOS/Linux
# =========================================================

echo
echo "🚀 Starting PubCrawler Web Interface for Network Access..."
echo

# Default port
PORT=""

# Parse port from command line arguments (if provided)
if [ -n "$1" ]; then
    PORT="$1"
else
    # Randomize port between 1024 and 65535
    PORT=$(jot -r 1 1024 65535)
fi

# Find Python command
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed or not in PATH"
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
    $PYTHON_CMD -m pip install -r requirements.txt || {
        echo "❌ Failed to install requirements"
        echo "💡 Try running: $PYTHON_CMD -m pip install --upgrade pip"
        exit 1
    }
fi

# Start ngrok with custom domain and selected port
echo "🌐 Starting ngrok tunnel on port $PORT..."
ngrok http --domain=evident-fawn-modest.ngrok-free.app $PORT &

# Start the web server
IP=$(ipconfig getifaddr en0 2>/dev/null || echo "localhost")
echo "✅ Your network URL is: http://$IP:$PORT"
echo
echo "💡 Tips:"
echo "   • Share the network URL above with your coworkers"
echo "   • Use Ctrl+C to stop the server"
echo "   • Check WEB_README.md for more information"
echo

# Add --no-reload to prevent double startup (if supported)
$PYTHON_CMD run_web.py --host 0.0.0.0 --port $PORT --debug --no-reload