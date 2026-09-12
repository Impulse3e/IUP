from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from server.app.backup import backup_database
from server.app.cleanup import run_cleanup
from server.app.config import settings
from server.app.database import SessionLocal
from server.app.models import ExamSession, Violation
from server.app.services import risk_delta
from server.app.websocket import manager
from shared.constants import SessionStatus, ViolationType


def _stale_cutoff() -> datetime:
    return datetime.utcnow() - timedelta(seconds=settings.heartbeat_timeout_sec)


def scan_lost_heartbeats(db: Session) -> list[tuple[str, dict]]:
    cutoff = _stale_cutoff()
    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.status == SessionStatus.ACTIVE.value)
        .filter(ExamSession.last_heartbeat.isnot(None))
        .filter(ExamSession.last_heartbeat < cutoff)
        .all()
    )
    emitted: list[tuple[str, dict]] = []
    for session in sessions:
        recent = (
            db.query(Violation)
            .filter(Violation.session_id == session.id, Violation.type == ViolationType.HEARTBEAT_LOST.value)
            .order_by(Violation.created_at.desc())
            .first()
        )
        if recent and not recent.is_resolved:
            continue
        violation = Violation(
            session_id=session.id,
            type=ViolationType.HEARTBEAT_LOST.value,
            message="агент не отвечает: пропал heartbeat.",
            severity="high",
            payload={"last_heartbeat": session.last_heartbeat.isoformat() if session.last_heartbeat else None},
        )
        session.risk_score = max(0.0, session.risk_score + risk_delta(ViolationType.HEARTBEAT_LOST.value, "high", False))
        db.add(violation)
        db.flush()
        emitted.append(
            (
                session.id,
                {
                    "id": violation.id,
                    "session_id": session.id,
                    "type": violation.type,
                    "message": violation.message,
                    "severity": violation.severity,
                    "payload": violation.payload,
                    "is_reminder": False,
                    "is_resolved": False,
                    "created_at": violation.created_at.isoformat() if violation.created_at else datetime.utcnow().isoformat(),
                },
            )
        )
    return emitted


def resolve_heartbeat_if_needed(db: Session, session: ExamSession) -> dict | None:
    recent = (
        db.query(Violation)
        .filter(Violation.session_id == session.id, Violation.type == ViolationType.HEARTBEAT_LOST.value)
        .order_by(Violation.created_at.desc())
        .first()
    )
    if not recent or recent.is_resolved:
        return None
    recovered = Violation(
        session_id=session.id,
        type=ViolationType.HEARTBEAT_LOST.value,
        message="Норма восстановлена: агент снова на связи.",
        severity="warning",
        payload={},
        is_resolved=True,
    )
    recent.is_resolved = True
    session.risk_score = max(0.0, session.risk_score + risk_delta(ViolationType.HEARTBEAT_LOST.value, "warning", True))
    db.add(recovered)
    db.flush()
    return {
        "id": recovered.id,
        "session_id": session.id,
        "type": recovered.type,
        "message": recovered.message,
        "severity": recovered.severity,
        "payload": recovered.payload,
        "is_reminder": False,
        "is_resolved": True,
        "created_at": recovered.created_at.isoformat() if recovered.created_at else datetime.utcnow().isoformat(),
    }


async def heartbeat_watchdog() -> None:
    ticks = 0
    last_backup_at = 0.0
    while True:
        await asyncio.sleep(settings.heartbeat_check_interval_sec)
        ticks += 1
        db = SessionLocal()
        try:
            events = scan_lost_heartbeats(db)
            if ticks == 1 or ticks % max(1, settings.cleanup_interval_sec // settings.heartbeat_check_interval_sec) == 0:
                run_cleanup(db)
            db.commit()
            now = datetime.utcnow().timestamp()
            if last_backup_at == 0.0 or now - last_backup_at >= 6 * 3600:
                try:
                    backup_database()
                    last_backup_at = now
                except Exception:
                    pass
            for session_id, payload in events:
                await manager.broadcast(session_id, {"type": "violation", "data": payload})
        except Exception:
            db.rollback()
        finally:
            db.close()
