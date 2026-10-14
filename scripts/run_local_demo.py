"""Run the public showcase locally in demo mode, without Docker.

    python scripts/run_local_demo.py            # http://127.0.0.1:8011/
    python scripts/run_local_demo.py --port 9000

Reads .env / .env.local for the OpenRouter key and the database URL, switches demo mode on
(uploads closed, questions budgeted, corpus seeded from demo/ when missing), and serves the
playground at "/". It does not read render.yaml, so production settings such as
ENVIRONMENT=production (which hides /docs) and TOP_K_RETRIEVAL=10 apply only if .env.local sets
them too.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    os.environ.setdefault("DEMO_MODE", "true")
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
