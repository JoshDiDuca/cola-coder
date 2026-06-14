#!/usr/bin/env python
"""Launch the cola-coder local web UI (dashboard over the CLI).

    python scripts/ui_server.py                # http://127.0.0.1:8800
    python scripts/ui_server.py --port 9000 --open

Read-only views (training status, GPU, checkpoints, datasets, scores) + a
background-job runner that drives the existing scripts. The server is local-only by
default; it never launches a second trainer (the UI refuses if one is running).
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cola_coder.cli import cli  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cola-Coder local web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args()

    try:
        import uvicorn
        from cola_coder.ui import create_app
    except ImportError as e:
        cli.error(f"Missing dependency: {e}. Install with: pip install -e \".[logging]\"")
        sys.exit(1)

    app = create_app(project_root=Path(__file__).resolve().parent.parent)
    url = f"http://{args.host}:{args.port}"
    cli.header("Cola-Coder UI")
    cli.success(f"Dashboard: {url}")
    cli.dim("Read-only + background jobs. Ctrl-C to stop. Training is never interrupted.")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
