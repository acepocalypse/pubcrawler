#!/usr/bin/env python3
"""
PubCrawler Web Server Launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple launcher script for the PubCrawler web interface.

Usage:
    python run_web.py [--host HOST] [--port PORT] [--debug]

Examples:
    python run_web.py                    # Run on localhost:5000
    python run_web.py --port 8080        # Run on localhost:8080
    python run_web.py --host 0.0.0.0     # Allow external connections
    python run_web.py --debug            # Enable debug mode
"""

import argparse
import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import flask
    except ImportError:
        missing_deps.append('flask')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import rapidfuzz
    except ImportError:
        missing_deps.append('rapidfuzz')
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\nInstall them with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch PubCrawler web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--host', 
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port', 
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no-check', 
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not args.no_check and not check_dependencies():
        sys.exit(1)
    
    # Import and run the Flask app
    try:
        from app import app
        
        print("🚀 Starting PubCrawler Web Interface...")
        print(f"   📍 URL: http://{args.host}:{args.port}")
        print(f"   🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
        print()
        print("💡 Tips:")
        print("   • Make sure to have API keys for Scopus/WoS for full functionality")
        print("   • Use Ctrl+C to stop the server")
        print()
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("Make sure you're in the correct directory and Flask is installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
