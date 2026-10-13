"""Run the live PostgreSQL + pgvector integration tests against an embedded server.

Useful on machines without Docker: ``pip install pgserver`` ships PostgreSQL binaries with the
pgvector extension inside the virtualenv, no admin rights required.

    python scripts/run_live_db_tests.py            # runs tests/test_integration_db.py
    python scripts/run_live_db_tests.py -k filter  # extra args are passed to pytest
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        import pgserver
    except ImportError:
        print("pgserver is not installed: pip install pgserver", file=sys.stderr)
        return 2
    import pytest

    data_dir = pathlib.Path(tempfile.mkdtemp(prefix="hybrid-rag-pg-"))
    server = pgserver.get_server(data_dir)
    try:
        uri = server.get_uri()
        async_url = uri.replace("postgresql://", "postgresql+asyncpg://", 1)
        os.environ["TEST_DATABASE_URL"] = async_url
        print(f"Embedded PostgreSQL ready: {uri}")
        os.chdir(ROOT)
        return int(pytest.main(["tests/test_integration_db.py", "-v", *sys.argv[1:]]))
    finally:
        # pytest closes the captured streams pgserver's logger writes to; silence it before stopping.
        logging.getLogger("pgserver").disabled = True
        server.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
