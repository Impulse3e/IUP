"""Unified entry point for IUP Student (launcher + proctoring agent)."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        if "--token" in sys.argv:
            from agent.main import main as run_agent
            from agent.ui import report_crash

            try:
                run_agent()
            except SystemExit:
                raise
            except Exception as error:
                report_crash(error)
            return

        from agent.launcher import main as run_launcher

        run_launcher()
    except SystemExit:
        raise
    except Exception as error:
        from agent.ui import report_crash

        report_crash(error)


if __name__ == "__main__":
    main()
