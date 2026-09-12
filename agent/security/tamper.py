import sys
import time


class TamperGuard:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.compromised = False

    def check_debugger(self) -> bool:
        return bool(sys.gettrace())

    def mark_compromised(self, reason: str) -> None:
        self.compromised = True
        self.reason = reason

    def status(self) -> dict:
        return {
            "compromised": self.compromised,
            "reason": getattr(self, "reason", ""),
            "uptime_sec": int(time.time() - self.started_at),
        }
