import hashlib
import hmac
import json
from typing import Any

import httpx
from sqlalchemy.orm import Session

from server.app.models import AuditLog, Webhook
from server.app.storage import sign_payload


def log_audit(db: Session, user_id: str | None, action: str, resource: str, details: dict | None = None) -> None:
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
        )
    )


async def dispatch_webhooks(db: Session, event_type: str, payload: dict[str, Any]) -> None:
    hooks = db.query(Webhook).filter(Webhook.active.is_(True)).all()
    body = {"event": event_type, "payload": payload}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for hook in hooks:
            if hook.events and event_type not in hook.events:
                continue
            headers = {"Content-Type": "application/json"}
            if hook.secret:
                headers["X-IUP-Signature"] = sign_payload(hook.secret, body)
            try:
                await client.post(hook.url, json=body, headers=headers)
            except Exception:
                continue


def risk_delta(event_type: str, severity: str, is_resolved: bool) -> float:
    if is_resolved:
        return -0.5
    weights = {
        "identity_mismatch": 25,
        "agent_tamper": 20,
        "forbidden_process": 15,
        "multiple_faces": 12,
        "no_face": 10,
        "look_away": 8,
        "window_focus_lost": 8,
        "second_monitor": 10,
        "audio_loud": 5,
        "audio_quiet": 4,
    }
    base = weights.get(event_type, 5)
    if severity == "critical":
        base *= 1.5
    return base
