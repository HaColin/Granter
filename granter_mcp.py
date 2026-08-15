"""Entry point for the MCP server, runnable from any working directory.

An MCP client starts its servers from whatever directory it happens to be in,
not from this one, so ``python -m granter.mcp_server`` fails for them with an
import error. This puts the project on the path first, so the command can be a
plain absolute path to this file:

    python C:\\path\\to\\Granter\\granter_mcp.py

The corpus location is derived from the package's own location, so it is found
regardless of where the process was started.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from granter.mcp_server import main  # noqa: E402  (path set up above)

if __name__ == "__main__":
    main()
