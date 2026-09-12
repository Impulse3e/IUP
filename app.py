"""Legacy entrypoint. Use: python -m agent.main"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "agent.main", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
