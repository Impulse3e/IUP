from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from server.app.config import settings
from server.app.models import Evidence, Exam, ExamSession, VideoChunk
from shared.constants import SessionStatus

KEEP_CHUNK_SOURCES = {"webcam": 4, "screen": 4, "preview": 1}
ENDED = {SessionStatus.COMPLETED.value, SessionStatus.CANCELLED.value, SessionStatus.COMPROMISED.value}


def _unlink(path: str) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def prune_old_chunks(db: Session) -> int:
    removed = 0
    session_ids = [row[0] for row in db.query(VideoChunk.session_id).distinct().all()]
    for session_id in session_ids:
        for source, keep in KEEP_CHUNK_SOURCES.items():
            rows = (
                db.query(VideoChunk)
                .filter(VideoChunk.session_id == session_id, VideoChunk.source == source)
                .order_by(VideoChunk.created_at.desc())
                .all()
            )
            for extra in rows[keep:]:
                _unlink(extra.path)
                db.delete(extra)
                removed += 1
    return removed


def prune_expired_sessions(db: Session) -> int:
    cutoff = datetime.utcnow() - timedelta(days=settings.retention_days)
    sessions = (
        db.query(ExamSession)
        .join(Exam, Exam.id == ExamSession.exam_id)
        .filter(ExamSession.status.in_(ENDED))
        .filter((ExamSession.ended_at != None) | (ExamSession.created_at != None))  # noqa: E711
        .all()
    )
    removed = 0
    for session in sessions:
        stamp = session.ended_at or session.created_at
        exam = db.get(Exam, session.exam_id)
        days = exam.retention_days if exam else settings.retention_days
        local_cutoff = datetime.utcnow() - timedelta(days=days)
        if not stamp or stamp > min(cutoff, local_cutoff):
            continue
        for chunk in db.query(VideoChunk).filter(VideoChunk.session_id == session.id).all():
            _unlink(chunk.path)
            db.delete(chunk)
            removed += 1
        for item in db.query(Evidence).filter(Evidence.session_id == session.id).all():
            _unlink(item.path)
            db.delete(item)
            removed += 1
    return removed


def run_cleanup(db: Session) -> dict[str, int]:
    chunks = prune_old_chunks(db)
    expired = prune_expired_sessions(db)
    return {"chunks": chunks, "expired_files": expired}
