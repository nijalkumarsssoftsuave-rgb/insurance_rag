"""Starts the API with an event loop psycopg can actually use.

Why this exists instead of plain ``uvicorn app.main:app``:

uvicorn creates its event loop *before* importing the application, so a policy
call inside ``app.db.session`` runs too late. On Windows the default is
``ProactorEventLoop``, which psycopg3 refuses to run async on - every database
call fails while Qdrant and the sync paths look healthy, which is a confusing
way to discover the problem.

On Linux and macOS this is an ordinary uvicorn launch with no special handling.

    python scripts/run_api.py
    python scripts/run_api.py --reload --port 8010
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Insurance RAG API")
    parser.add_argument("--host", default=settings.app.api_host)
    parser.add_argument("--port", type=int, default=settings.app.api_port)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=settings.app.log_level.lower())
    args = parser.parse_args()

    import uvicorn

    if sys.platform != "win32":
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
        return 0

    # Windows needs the loop built by hand. Current uvicorn selects its loop via
    # `uvicorn.loops.asyncio.asyncio_loop_factory`, which returns ProactorEventLoop
    # on win32 and ignores the event loop policy entirely - so setting the policy
    # (the usual advice) silently does nothing. Running the server ourselves on a
    # SelectorEventLoop is the only reliable fix.
    if args.reload:
        # Reload runs the server in a subprocess we do not control the loop of.
        print(
            "--reload cannot be combined with the Windows selector loop; "
            "async database calls would fail. Restart manually instead.",
            file=sys.stderr,
        )
        return 2

    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        loop="none",  # we supply it
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
