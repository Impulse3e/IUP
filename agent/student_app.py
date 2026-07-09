"""Unified entry point for IUP Student (launcher + proctoring agent)."""

from __future__ import annotations

import sys


def main() -> None:
    if "--token" in sys.argv:
        from agent.main import main as run_agent

        run_agent()
        return

    from agent.launcher import main as run_launcher

    run_launcher()


if __name__ == "__main__":
    main()
