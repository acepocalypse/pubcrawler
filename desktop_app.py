"""
PubCrawler Desktop App Launcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Launches the existing Flask app in a background thread and displays
the interface in a native desktop window using pywebview.

This entry point is intended for single-user, local desktop use.
It keeps the original network-accessible CLI entry (run_web.py) unchanged.

Build a standalone executable with PyInstaller; see README for details.
"""

import argparse
import os
import socket
import sys
import threading
import time
from contextlib import closing
from pathlib import Path


def _resource_path(*parts: str) -> str:
    """Return absolute path to bundled resources when frozen or fallback to repo paths."""
    # When packaged by PyInstaller, `_MEIPASS` points to the temp unpack directory (onefile)
    # or the app directory (onedir). Fallback to the directory of this file or CWD.
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return str(Path(base, *parts))
    # If running from a frozen executable, sys.executable is the binary path
    if getattr(sys, "frozen", False):  # type: ignore[attr-defined]
        return str(Path(Path(sys.executable).parent, *parts))
    # Dev/runtime from source
    return str(Path(__file__).parent.joinpath(*parts))


def _find_free_port(preferred=(5000, 5001, 5002, 5003, 5004)) -> int:
    """Find an available TCP port on localhost, preferring common defaults."""
    # Try preferred ports first
    for p in preferred:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    # Let OS pick a free ephemeral port
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 20.0, interval: float = 0.2) -> bool:
    """Poll the given URL until it responds or times out."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:  # consider up even if 404 (route mismatch)
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(interval)
    return False


def _configure_template_static_paths(app_obj):
    """Point Flask to bundled templates/static when running from a frozen build."""
    try:
        from jinja2 import FileSystemLoader
    except Exception:
        FileSystemLoader = None  # type: ignore

    templates_dir = Path(_resource_path("templates"))
    static_dir = Path(_resource_path("static"))

    if templates_dir.exists():
        app_obj.template_folder = str(templates_dir)
        if FileSystemLoader is not None:
            app_obj.jinja_loader = FileSystemLoader(str(templates_dir))
    if static_dir.exists():
        app_obj.static_folder = str(static_dir)


def start_flask_in_thread(host: str, port: int, debug: bool = False):
    """Import the existing Flask app and run it in a daemon thread."""
    from app import app as flask_app  # import here so failures show helpful error

    # When packaged, ensure Flask can locate templates/static
    _configure_template_static_paths(flask_app)

    def _run():
        # Disable reloader to avoid double startups in a thread
        flask_app.run(host=host, port=port, debug=debug, use_reloader=False)

    t = threading.Thread(target=_run, name="FlaskServer", daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser(description="PubCrawler Desktop App")
    parser.add_argument("--port", type=int, default=None, help="Port for local server (default: auto)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug (for development)")
    args = parser.parse_args()

    # Choose host/port
    host = "127.0.0.1"
    port = args.port or _find_free_port()

    # Start Flask
    try:
        server_thread = start_flask_in_thread(host, port, debug=args.debug)
    except Exception as e:
        print("Failed to start Flask server:", e)
        sys.exit(1)

    url = f"http://{host}:{port}/"
    # Wait until server responds
    if not _wait_for_server(url, timeout=30.0):
        print("Timed out waiting for local server at", url)
        sys.exit(1)

    # Launch desktop window
    try:
        import webview
    except Exception as e:
        print("pywebview is required for the desktop app but was not found.")
        print("Install it with: pip install pywebview")
        sys.exit(1)

    # Window title and optional icon (supported on newer pywebview only)
    title = "PubCrawler"
    icon_path = _resource_path("favicon.ico")
    icon = icon_path if os.path.exists(icon_path) else None

    import inspect
    sig = inspect.signature(webview.create_window)
    common_kwargs = dict(
        width=1200,
        height=800,
        resizable=True,
        confirm_close=False,
        minimized=False,
        background_color="#ffffff",
        text_select=True,
    )
    if "icon" in sig.parameters and icon:
        common_kwargs["icon"] = icon

    window = webview.create_window(title, url, **common_kwargs)

    # Start GUI loop (blocks until window closed)
    try:
        webview.start()
    except Exception as e:
        print("Failed to start webview:", e)
        sys.exit(1)

    # When window closes, process will exit; Flask runs in daemon thread
    # Optionally wait briefly for cleanup
    time.sleep(0.2)


if __name__ == "__main__":
    main()
