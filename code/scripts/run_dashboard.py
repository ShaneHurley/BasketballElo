#!/usr/bin/env python3
"""Launch the local T-60 Analysis Dashboard on 127.0.0.1:8765 by default."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure BasketballElo/code is on sys.path when run as a script.
CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def main() -> None:
    from dashboard.config import DEFAULT_HOST, DEFAULT_PORT

    p = argparse.ArgumentParser(description="T-60 local analysis dashboard")
    p.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default localhost)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = p.parse_args()
    if args.host not in ("127.0.0.1", "localhost") and args.host != DEFAULT_HOST:
        print(f"WARNING: binding {args.host} exposes the dashboard on the network.")
    import uvicorn
    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
