"""
app.py -- launch the review app as a native macOS window (pywebview), backed by
the FastAPI server. Falls back to a browser tab if pywebview isn't installed.

    cd sim_cut && python -m ui.app
"""
from __future__ import annotations

import socket
import threading
import time

import uvicorn

from .server import app as fastapi_app


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(port):
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, log_level="warning")


class Api:
    """Exposed to JS as window.pywebview.api -- the native folder picker."""

    def pick_folder(self):
        import webview
        folder = getattr(getattr(webview, "FileDialog", None), "FOLDER", None)
        if folder is None:
            folder = webview.FOLDER_DIALOG               # older pywebview
        res = webview.windows[0].create_file_dialog(folder)
        if not res:
            return ""
        return res[0] if isinstance(res, (list, tuple)) else res


def main():
    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):                       # wait until the server accepts connections
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            break
        except OSError:
            time.sleep(0.1)

    try:
        import webview
        webview.create_window("sim_cut — review", url, width=1180, height=820,
                              min_size=(900, 640), js_api=Api())
        webview.start()
    except Exception as e:  # noqa: BLE001
        import webbrowser
        print(f"[sim_cut] pywebview unavailable ({e}); opening in browser: {url}")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
