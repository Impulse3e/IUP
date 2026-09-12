from dataclasses import dataclass, field
from typing import Any

from shared.constants import ViolationType


@dataclass
class ProctorEvent:
    type: str
    message: str
    severity: str = "warning"
    payload: dict[str, Any] = field(default_factory=dict)
    is_reminder: bool = False
    is_resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "severity": self.severity,
            "payload": self.payload,
            "is_reminder": self.is_reminder,
            "is_resolved": self.is_resolved,
        }


@dataclass
class SessionSummary:
    duration_sec: int
    violation_totals: dict[str, int]
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_sec": self.duration_sec,
            "violation_totals": self.violation_totals,
            "risk_score": self.risk_score,
        }
