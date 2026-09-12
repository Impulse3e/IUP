import asyncio
import csv
import hmac
import html
import io
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import and_, case, func
from sqlalchemy.orm.attributes import flag_modified

from server.app.auth import create_access_token, get_current_user, get_user_from_token, hash_password, require_roles, verify_password
from server.app.config import settings
from server.app.database import get_db
from server.app.models import Exam, ExamSession, Evidence, Review, User, VideoChunk, Violation, Webhook
from server.app.schemas import (
    ConsentRequest,
    EventRequest,
    EvidenceResponse,
    ExamCreate,
    ExamCreateResponse,
    ExamResponse,
    ExportResponse,
    HeartbeatRequest,
    IdentityRequest,
    IdentityVerifyResponse,
    LTILaunchRequest,
    ReviewCreate,
    ReviewResponse,
    SessionAssignResponse,
    SessionCreate,
    SessionCreateByEmail,
    SessionEndRequest,
    SessionResponse,
    SessionWithExamResponse,
    StudentSessionResponse,
    TokenResponse,
    UserCreate,
    UserResponse,
    ViolationResponse,
    WebhookCreate,
    WebhookResponse,
)
from server.app.security import (
    ALLOWED_CHUNK_SOURCES,
    STAFF_ROLES,
    generate_temp_password,
    upload_suffix,
)
from server.app.identity import embedding_distance
from server.app.backup import backup_database
from server.app.exam_window import exam_phase, exam_window_message
from server.app.services import dispatch_webhooks, log_audit, risk_delta
from server.app.storage import get_storage
from server.app.watchdog import resolve_heartbeat_if_needed
from server.app.websocket import manager
from shared.constants import SessionStatus, ViolationType

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _agent_install_hint() -> str:
    env = os.getenv("IUP_INSTALL_PATH")
    if env:
        return env
    return str(PROJECT_ROOT)


LIVE_SESSION_STATUSES = {SessionStatus.ACTIVE.value, SessionStatus.PRECHECK.value}
CRITICAL_OPEN_TYPES = {
    ViolationType.IDENTITY_MISMATCH.value,
    ViolationType.AGENT_TAMPER.value,
    ViolationType.FORBIDDEN_PROCESS.value,
    ViolationType.MULTIPLE_FACES.value,
    ViolationType.HEARTBEAT_LOST.value,
}


def _session_is_online(session: ExamSession) -> bool:
    if session.status not in LIVE_SESSION_STATUSES or not session.last_heartbeat:
        return False
    age = (datetime.utcnow() - session.last_heartbeat).total_seconds()
    return age <= settings.heartbeat_timeout_sec + 8


def _event_stats_by_session(db: Session, session_ids: list[str]) -> dict[str, tuple[int, int]]:
    if not session_ids:
        return {}
    rows = (
        db.query(
            Violation.session_id,
            func.sum(
                case(
                    (
                        and_(
                            Violation.is_resolved.is_(False),
                            Violation.is_reminder.is_(False),
                            Violation.type.in_(CRITICAL_OPEN_TYPES),
                        ),
                        1,
                    ),
                    else_=0,
                )
            ),
            func.sum(
                case(
                    (
                        and_(Violation.is_resolved.is_(False), Violation.is_reminder.is_(False)),
                        1,
                    ),
                    else_=0,
                )
            ),
        )
        .filter(Violation.session_id.in_(session_ids))
        .group_by(Violation.session_id)
        .all()
    )
    return {row[0]: (int(row[1] or 0), int(row[2] or 0)) for row in rows}


ENDED_SESSION_STATUSES = {SessionStatus.COMPLETED.value, SessionStatus.CANCELLED.value}


def _session_by_token(db: Session, token: str) -> ExamSession:
    session = db.query(ExamSession).filter(ExamSession.access_token == token).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _exam_for(db: Session, session: ExamSession) -> Exam | None:
    return db.get(Exam, session.exam_id)


def _reject_if_ended(session: ExamSession) -> None:
    if session.status in ENDED_SESSION_STATUSES:
        raise HTTPException(status_code=409, detail="Сессия завершена")


def _reject_if_exam_not_open(db: Session, session: ExamSession) -> None:
    exam = _exam_for(db, session)
    if not exam:
        return
    message = exam_window_message(exam)
    if message:
        raise HTTPException(status_code=403, detail=message)


def _reject_if_cannot_record(db: Session, session: ExamSession) -> None:
    _reject_if_ended(session)
    exam = _exam_for(db, session)
    if not exam:
        return
    if exam_phase(exam) == "after":
        raise HTTPException(status_code=409, detail="Окно экзамена закрыто")
    if exam_phase(exam) == "before":
        raise HTTPException(status_code=403, detail="Экзамен ещё не начался")


def _session_public(db: Session, session: ExamSession) -> dict:
    exam = db.get(Exam, session.exam_id)
    student = db.get(User, session.student_id)
    data = SessionResponse.model_validate(session).model_dump()
    data["exam_title"] = exam.title if exam else ""
    data["student_name"] = student.full_name if student else ""
    data["student_email"] = student.email if student else ""
    data["online"] = _session_is_online(session)
    settings = exam.settings if exam else {}
    data["exam_opens_at"] = str(settings.get("opens_at") or "")
    data["exam_closes_at"] = str(settings.get("closes_at") or "")
    data["critical_open"] = 0
    data["open_events"] = 0
    return data


def _session_detail(db: Session, session: ExamSession) -> SessionWithExamResponse:
    data = _session_public(db, session)
    crit, open_events = _event_stats_by_session(db, [session.id]).get(session.id, (0, 0))
    data["critical_open"] = crit
    data["open_events"] = open_events
    return SessionWithExamResponse(**data)


def _student_session_detail(db: Session, session: ExamSession) -> StudentSessionResponse:
    data = _session_public(db, session)
    data["access_token"] = session.access_token
    return StudentSessionResponse(**data)


def _get_or_create_student(db: Session, email: str, full_name: str) -> tuple[User, str | None]:
    student = db.query(User).filter(User.email == email).first()
    if student:
        return student, None
    password = generate_temp_password()
    student = User(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name or email.split("@")[0],
        role="student",
    )
    db.add(student)
    db.flush()
    return student, password


async def _enforce_upload_limit(request: Request) -> None:
    length = request.headers.get("content-length")
    if not length:
        return
    try:
        size = int(length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    if size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")


def _ws_bearer_token(websocket: WebSocket, token: str | None) -> str | None:
    if token:
        return token
    header = websocket.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()
    return None


def _safe_stored_file(stored_path: str) -> Path:
    root = Path(settings.storage_path).resolve()
    path = Path(stored_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return path


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".avi": "video/x-msvideo",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".bin": "image/jpeg",
    }.get(suffix, "application/octet-stream")


def _evidence_response(session_id: str, item: Evidence) -> EvidenceResponse:
    return EvidenceResponse(
        id=item.id,
        session_id=item.session_id,
        violation_id=item.violation_id,
        type=item.type,
        created_at=item.created_at,
        url=f"/api/sessions/{session_id}/evidence/{item.id}/file",
    )


@router.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/auth/register", response_model=UserResponse)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles("admin")),
):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    db.add(user)
    log_audit(db, admin.id, "user.create", f"user:{payload.email}")
    db.commit()
    db.refresh(user)
    return user


@router.get("/users/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return user


@router.post("/exams", response_model=ExamCreateResponse)
def create_exam(
    payload: ExamCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "admin")),
):
    exam_settings = payload.settings or {}
    exam_settings.setdefault("open_enrollment", True)
    exam = Exam(
        title=payload.title,
        description=payload.description,
        created_by=user.id,
        settings=exam_settings,
        retention_days=payload.retention_days,
    )
    db.add(exam)
    db.flush()
    log_audit(db, user.id, "exam.create", f"exam:{exam.id}")

    initial_password = None
    created_student_email = None
    if payload.student_email.strip():
        email = payload.student_email.strip().lower()
        student, initial_password = _get_or_create_student(db, email, payload.student_name)
        created_student_email = student.email if initial_password else None
        session = ExamSession(
            exam_id=exam.id,
            student_id=student.id,
            access_token=secrets.token_urlsafe(32),
            status=SessionStatus.PENDING.value,
        )
        db.add(session)
        log_audit(db, user.id, "session.create", f"session:{session.id}")

    db.commit()
    db.refresh(exam)
    data = ExamResponse.model_validate(exam).model_dump()
    data["initial_password"] = initial_password
    data["created_student_email"] = created_student_email
    return ExamCreateResponse(**data)


@router.get("/exams", response_model=list[ExamResponse])
def list_exams(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if user.role in {"teacher", "admin"}:
        return db.query(Exam).order_by(Exam.created_at.desc()).all()
    session_exam_ids = {s.exam_id for s in db.query(ExamSession).filter(ExamSession.student_id == user.id).all()}
    return db.query(Exam).filter(Exam.id.in_(session_exam_ids)).all()


@router.post("/exams/{exam_id}/sessions", response_model=SessionResponse)
def create_session(
    exam_id: str,
    payload: SessionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "proctor", "admin")),
):
    exam = db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    student = db.get(User, payload.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    session = ExamSession(
        exam_id=exam_id,
        student_id=payload.student_id,
        access_token=secrets.token_urlsafe(32),
        status=SessionStatus.PENDING.value,
    )
    db.add(session)
    log_audit(db, user.id, "session.create", f"session:{session.id}")
    db.commit()
    db.refresh(session)
    return session


@router.get("/students", response_model=list[UserResponse])
def list_students(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "proctor", "admin")),
):
    return db.query(User).filter(User.role == "student").order_by(User.full_name).all()


@router.get("/my/sessions", response_model=list[StudentSessionResponse])
def my_sessions(db: Session = Depends(get_db), user: User = Depends(require_roles("student"))):
    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.student_id == user.id)
        .order_by(ExamSession.created_at.desc())
        .all()
    )
    return [_student_session_detail(db, session) for session in sessions]


ACTIVE_SESSION_STATUSES = {
    SessionStatus.PENDING.value,
    SessionStatus.PRECHECK.value,
    SessionStatus.ACTIVE.value,
}


@router.get("/my/available-exams", response_model=list[ExamResponse])
def available_exams(db: Session = Depends(get_db), user: User = Depends(require_roles("student"))):
    active_exam_ids = {
        session.exam_id
        for session in db.query(ExamSession)
        .filter(
            ExamSession.student_id == user.id,
            ExamSession.status.in_(ACTIVE_SESSION_STATUSES),
        )
        .all()
    }
    exams = db.query(Exam).order_by(Exam.created_at.desc()).all()
    result = []
    for exam in exams:
        exam_settings = exam.settings or {}
        if exam_settings.get("open_enrollment", True) is False:
            continue
        if exam.id in active_exam_ids:
            continue
        if exam_phase(exam) == "after":
            continue
        result.append(exam)
    return result


@router.post("/exams/{exam_id}/join", response_model=StudentSessionResponse)
def join_exam(
    exam_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    exam = db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    exam_settings = exam.settings or {}
    if exam_settings.get("open_enrollment", True) is False:
        raise HTTPException(status_code=403, detail="Экзамен доступен только по приглашению")
    message = exam_window_message(exam)
    if message:
        raise HTTPException(status_code=403, detail=message)

    active = (
        db.query(ExamSession)
        .filter(
            ExamSession.exam_id == exam_id,
            ExamSession.student_id == user.id,
            ExamSession.status.in_(ACTIVE_SESSION_STATUSES),
        )
        .first()
    )
    if active:
        return _student_session_detail(db, active)

    session = ExamSession(
        exam_id=exam_id,
        student_id=user.id,
        access_token=secrets.token_urlsafe(32),
        status=SessionStatus.PENDING.value,
    )
    db.add(session)
    log_audit(db, user.id, "session.join", f"session:{session.id}")
    db.commit()
    db.refresh(session)
    return _student_session_detail(db, session)


@router.get("/exams/{exam_id}/sessions", response_model=list[SessionWithExamResponse])
def exam_sessions(
    exam_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "proctor", "admin")),
):
    exam = db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.exam_id == exam_id)
        .order_by(ExamSession.created_at.desc())
        .all()
    )
    return [_session_detail(db, session) for session in sessions]


@router.post("/exams/{exam_id}/sessions/by-email", response_model=SessionAssignResponse)
def create_session_by_email(
    exam_id: str,
    payload: SessionCreateByEmail,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "proctor", "admin")),
):
    exam = db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    email = payload.email.strip().lower()
    student, initial_password = _get_or_create_student(db, email, payload.full_name)
    session = ExamSession(
        exam_id=exam_id,
        student_id=student.id,
        access_token=secrets.token_urlsafe(32),
        status=SessionStatus.PENDING.value,
    )
    db.add(session)
    log_audit(db, user.id, "session.create", f"session:{session.id}")
    db.commit()
    db.refresh(session)
    data = _session_detail(db, session).model_dump()
    data["initial_password"] = initial_password
    return SessionAssignResponse(**data)


@router.get("/my/sessions/{session_id}/launch-info")
def session_launch_info(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    session = db.get(ExamSession, session_id)
    if not session or session.student_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    detail = _session_detail(db, session)
    return {
        "session_id": session.id,
        "access_token": session.access_token,
        "server_url": str(request.base_url).rstrip("/"),
        "exam_title": detail.exam_title,
        "status": session.status,
    }


@router.get("/my/sessions/{session_id}/launcher.sh", response_class=PlainTextResponse)
def download_launcher_script(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    session = db.get(ExamSession, session_id)
    if not session or session.student_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    server = str(request.base_url).rstrip("/")
    install = os.getenv("IUP_INSTALL_PATH") or str(PROJECT_ROOT)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
SERVER="{server}"
TOKEN="{session.access_token}"
if [ -n "${{IUP_INSTALL_PATH:-}}" ] && [ -x "$IUP_INSTALL_PATH/.venv/bin/python" ]; then
  ROOT="$IUP_INSTALL_PATH"
elif [ -x "{install}/.venv/bin/python" ]; then
  ROOT="{install}"
elif [ -x "$HOME/IUP/.venv/bin/python" ]; then
  ROOT="$HOME/IUP"
else
  echo "Не найден IUP. Запустите ./scripts/run_student.sh из папки проекта."
  read -r _
  exit 1
fi
cd "$ROOT"
export PYTHONPATH=.
exec "$ROOT/.venv/bin/python" -m agent.main --token "$TOKEN" --server "$SERVER" --consent-accepted
"""
    return PlainTextResponse(script, media_type="text/x-shellscript", headers={
        "Content-Disposition": f'attachment; filename="iup-exam-{session_id[:8]}.sh"'
    })


@router.get("/my/sessions/{session_id}/launcher.bat", response_class=PlainTextResponse)
def download_launcher_bat(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    session = db.get(ExamSession, session_id)
    if not session or session.student_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    server = str(request.base_url).rstrip("/")
    hint = _agent_install_hint().replace("\\", "\\\\")
    script = f"""@echo off
setlocal EnableExtensions
set "SERVER={server}"
set "TOKEN={session.access_token}"
set "HINT={hint}"

if defined IUP_INSTALL_PATH if exist "%IUP_INSTALL_PATH%\\.venv\\Scripts\\python.exe" (
  set "ROOT=%IUP_INSTALL_PATH%"
  goto :run
)
if exist "%HINT%\\.venv\\Scripts\\python.exe" (
  set "ROOT=%HINT%"
  goto :run
)
if exist "%USERPROFILE%\\IUP\\.venv\\Scripts\\python.exe" (
  set "ROOT=%USERPROFILE%\\IUP"
  goto :run
)

echo Не найден IUP Student.
echo Запустите scripts\\run_student.bat из папки проекта
echo или установите: powershell -ExecutionPolicy Bypass -File scripts\\install_windows.ps1
pause
exit /b 1

:run
cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
"%ROOT%\\.venv\\Scripts\\python.exe" -m agent.main --token "%TOKEN%" --server "%SERVER%" --consent-accepted
if errorlevel 1 pause
"""
    return PlainTextResponse(script, media_type="application/x-bat", headers={
        "Content-Disposition": f'attachment; filename="iup-exam-{session_id[:8]}.bat"'
    })


def _sessions_detail(db: Session, sessions: list[ExamSession]) -> list[SessionWithExamResponse]:
    if not sessions:
        return []
    exams = {
        exam.id: exam
        for exam in db.query(Exam).filter(Exam.id.in_({item.exam_id for item in sessions})).all()
    }
    students = {
        user.id: user
        for user in db.query(User).filter(User.id.in_({item.student_id for item in sessions})).all()
    }
    result = []
    stats = _event_stats_by_session(db, [item.id for item in sessions])
    for session in sessions:
        exam = exams.get(session.exam_id)
        student = students.get(session.student_id)
        data = SessionResponse.model_validate(session).model_dump()
        data["exam_title"] = exam.title if exam else ""
        data["student_name"] = student.full_name if student else ""
        data["student_email"] = student.email if student else ""
        data["online"] = _session_is_online(session)
        settings = exam.settings if exam else {}
        data["exam_opens_at"] = str(settings.get("opens_at") or "")
        data["exam_closes_at"] = str(settings.get("closes_at") or "")
        crit, open_events = stats.get(session.id, (0, 0))
        data["critical_open"] = crit
        data["open_events"] = open_events
        result.append(SessionWithExamResponse(**data))
    return result


@router.get("/sessions", response_model=list[SessionWithExamResponse])
def list_sessions(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    sessions = db.query(ExamSession).order_by(ExamSession.created_at.desc()).limit(200).all()
    return _sessions_detail(db, sessions)


@router.get("/sessions/token/{token}", response_model=SessionResponse)
def get_session_by_token(token: str, db: Session = Depends(get_db)):
    return _session_by_token(db, token)


@router.post("/sessions/token/{token}/consent")
async def accept_consent(token: str, payload: ConsentRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    _reject_if_ended(session)
    if not payload.accepted:
        session.status = SessionStatus.CANCELLED.value
    else:
        _reject_if_exam_not_open(db, session)
        if not session.consent_at:
            session.consent_at = datetime.utcnow()
        if session.status == SessionStatus.PENDING.value:
            session.status = SessionStatus.PRECHECK.value
    db.commit()
    await manager.broadcast(session.id, {"type": "consent", "session_id": session.id})
    return {"status": session.status}


@router.post("/sessions/token/{token}/identity", response_model=IdentityVerifyResponse)
async def verify_identity(token: str, payload: IdentityRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    _reject_if_ended(session)
    _reject_if_exam_not_open(db, session)
    student = db.get(User, session.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    stored = (student.face_embedding or {}).get("vector") if isinstance(student.face_embedding, dict) else None
    enrolled = bool(stored)
    distance = embedding_distance(stored, payload.embedding) if enrolled else None
    matched = None
    recent_mismatch = (
        db.query(Violation)
        .filter(Violation.session_id == session.id, Violation.type == ViolationType.IDENTITY_MISMATCH.value)
        .order_by(Violation.created_at.desc())
        .first()
    )
    if enrolled and distance is not None:
        matched = distance <= settings.identity_distance_threshold
        if not matched and (not recent_mismatch or recent_mismatch.is_resolved):
            violation = Violation(
                session_id=session.id,
                type=ViolationType.IDENTITY_MISMATCH.value,
                message=f"лицо не совпадает с эталоном на сервере (distance {distance:.3f}).",
                severity="critical",
                payload={"distance": distance, "threshold": settings.identity_distance_threshold},
            )
            session.risk_score = max(
                0.0,
                session.risk_score + risk_delta(ViolationType.IDENTITY_MISMATCH.value, "critical", False),
            )
            db.add(violation)
            db.flush()
            await manager.broadcast(
                session.id,
                {"type": "violation", "data": ViolationResponse.model_validate(violation).model_dump(mode="json")},
            )
        elif matched and recent_mismatch and not recent_mismatch.is_resolved:
            recent_mismatch.is_resolved = True
    else:
        student.face_embedding = {"vector": payload.embedding}
    session.identity_verified = True
    db.commit()
    await manager.broadcast(
        session.id,
        {"type": "identity_verified", "session_id": session.id, "match": matched, "distance": distance},
    )
    return IdentityVerifyResponse(
        identity_verified=True,
        enrolled=enrolled,
        match=matched,
        distance=distance,
    )


@router.post("/sessions/token/{token}/start", response_model=SessionResponse)
async def start_session(token: str, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    _reject_if_ended(session)
    if session.status == SessionStatus.ACTIVE.value:
        return session
    if not session.consent_at:
        raise HTTPException(status_code=400, detail="Consent required")
    if not session.identity_verified:
        raise HTTPException(status_code=400, detail="Identity verification required")
    _reject_if_exam_not_open(db, session)
    session.status = SessionStatus.ACTIVE.value
    session.started_at = session.started_at or datetime.utcnow()
    db.commit()
    db.refresh(session)
    await manager.broadcast_all({"type": "session_started", "session": SessionResponse.model_validate(session).model_dump(mode="json")})
    await dispatch_webhooks(db, "session.started", {"session_id": session.id})
    return session


@router.post("/sessions/token/{token}/heartbeat")
async def heartbeat(token: str, payload: HeartbeatRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    if session.status in ENDED_SESSION_STATUSES:
        return {"ok": False, "reason": "ended"}
    recovered = resolve_heartbeat_if_needed(db, session)
    session.last_heartbeat = datetime.utcnow()
    session.agent_version = payload.agent_version
    if payload.status == "compromised":
        session.status = SessionStatus.COMPROMISED.value
    db.commit()
    await manager.broadcast(session.id, {"type": "heartbeat", "session_id": session.id, "payload": payload.model_dump()})
    if recovered:
        await manager.broadcast(session.id, {"type": "violation", "data": recovered})
    return {"ok": True}


@router.post("/sessions/token/{token}/events", response_model=ViolationResponse)
async def post_event(token: str, payload: EventRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    _reject_if_cannot_record(db, session)
    violation = Violation(
        session_id=session.id,
        type=payload.type,
        message=payload.message,
        severity=payload.severity,
        payload=payload.payload,
        is_reminder=payload.is_reminder,
        is_resolved=payload.is_resolved,
    )
    session.risk_score = max(0.0, session.risk_score + risk_delta(payload.type, payload.severity, payload.is_resolved))
    db.add(violation)
    db.commit()
    db.refresh(violation)
    event_data = ViolationResponse.model_validate(violation).model_dump(mode="json")
    await manager.broadcast(session.id, {"type": "violation", "data": event_data})
    await dispatch_webhooks(db, "violation.created", event_data)
    return violation


@router.post("/sessions/token/{token}/end", response_model=SessionResponse)
async def end_session(token: str, payload: SessionEndRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    if session.status not in ENDED_SESSION_STATUSES:
        session.status = SessionStatus.COMPLETED.value
        session.ended_at = datetime.utcnow()
        summary = dict(session.summary or {})
        summary.update(payload.summary or {})
        session.summary = summary
        flag_modified(session, "summary")
    db.commit()
    db.refresh(session)
    await manager.broadcast_all({"type": "session_ended", "session_id": session.id})
    await dispatch_webhooks(db, "session.ended", {"session_id": session.id, "summary": payload.summary})
    return session


@router.post("/sessions/token/{token}/evidence")
async def upload_evidence(
    token: str,
    evidence_type: Annotated[str, Form()],
    file: UploadFile = File(...),
    violation_id: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
    _: None = Depends(_enforce_upload_limit),
):
    session = _session_by_token(db, token)
    _reject_if_cannot_record(db, session)
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    relative = f"sessions/{session.id}/evidence/{uuid.uuid4()}{upload_suffix(file.filename)}"
    storage = get_storage()
    path = storage.save_bytes(relative, content)
    evidence = Evidence(
        session_id=session.id,
        violation_id=violation_id,
        type=evidence_type,
        path=path,
    )
    db.add(evidence)
    db.commit()
    db.refresh(evidence)
    await manager.broadcast(session.id, {"type": "evidence", "evidence_id": evidence.id, "path": path})
    return {"id": evidence.id, "path": path}


@router.get("/sessions/{session_id}/evidence", response_model=list[EvidenceResponse])
def list_evidence(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    items = (
        db.query(Evidence)
        .filter(Evidence.session_id == session_id)
        .order_by(Evidence.created_at.desc())
        .all()
    )
    return [_evidence_response(session_id, item) for item in items]


@router.get("/sessions/{session_id}/evidence/{evidence_id}/file")
def download_evidence(
    session_id: str,
    evidence_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    item = db.query(Evidence).filter(Evidence.id == evidence_id, Evidence.session_id == session_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Evidence not found")
    path = _safe_stored_file(item.path)
    return FileResponse(path, media_type=_media_type_for(path), filename=path.name)


@router.get("/sessions/{session_id}/live-frame")
def live_frame(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    chunk = (
        db.query(VideoChunk)
        .filter(VideoChunk.session_id == session_id, VideoChunk.source == "preview")
        .order_by(VideoChunk.created_at.desc())
        .first()
    )
    if not chunk:
        chunk = (
            db.query(VideoChunk)
            .filter(VideoChunk.session_id == session_id, VideoChunk.source == "webcam")
            .order_by(VideoChunk.created_at.desc())
            .first()
        )
    if not chunk:
        raise HTTPException(status_code=404, detail="No live frame yet")
    path = _safe_stored_file(chunk.path)
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.post("/sessions/token/{token}/chunks")
async def upload_chunk(
    token: str,
    source: Annotated[str, Form()],
    chunk_index: Annotated[int, Form()],
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(_enforce_upload_limit),
):
    if source not in ALLOWED_CHUNK_SOURCES:
        raise HTTPException(status_code=400, detail="Invalid chunk source")
    if chunk_index < 0:
        raise HTTPException(status_code=400, detail="Invalid chunk index")
    session = _session_by_token(db, token)
    _reject_if_cannot_record(db, session)
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    if source == "preview":
        relative = f"sessions/{session.id}/live.jpg"
    else:
        relative = f"sessions/{session.id}/chunks/{source}_{chunk_index:06d}.bin"
    path = get_storage().save_bytes(relative, content)
    if source == "preview":
        chunk = (
            db.query(VideoChunk)
            .filter(VideoChunk.session_id == session.id, VideoChunk.source == "preview")
            .first()
        )
        if chunk:
            chunk.path = path
            chunk.chunk_index = chunk_index
            chunk.created_at = datetime.utcnow()
        else:
            db.add(VideoChunk(session_id=session.id, source=source, chunk_index=chunk_index, path=path))
    else:
        db.add(VideoChunk(session_id=session.id, source=source, chunk_index=chunk_index, path=path))
    db.commit()
    return {"path": path}


@router.get("/sessions/{session_id}/violations", response_model=list[ViolationResponse])
def session_violations(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    items = (
        db.query(Violation)
        .filter(Violation.session_id == session_id)
        .order_by(Violation.created_at.desc())
        .limit(200)
        .all()
    )
    return list(reversed(items))


ALLOWED_REVIEW_DECISIONS = {"confirmed", "false_positive", "invalidate", "passed"}


@router.post("/reviews", response_model=ReviewResponse)
def create_review(
    payload: ReviewCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    if payload.decision not in ALLOWED_REVIEW_DECISIONS:
        raise HTTPException(status_code=400, detail="Unknown decision")
    session = db.get(ExamSession, payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if payload.decision in {"passed", "invalidate"} and not payload.violation_id:
        summary = dict(session.summary or {})
        summary["outcome"] = payload.decision
        summary["outcome_comment"] = payload.comment or ""
        summary["reviewed_at"] = datetime.utcnow().isoformat()
        summary["reviewer_id"] = user.id
        session.summary = summary
        flag_modified(session, "summary")
        if payload.decision == "invalidate" and session.status not in ENDED_SESSION_STATUSES:
            session.status = SessionStatus.CANCELLED.value
            session.ended_at = datetime.utcnow()
    review = Review(
        session_id=payload.session_id,
        violation_id=payload.violation_id,
        reviewer_id=user.id,
        decision=payload.decision,
        comment=payload.comment,
    )
    db.add(review)
    log_audit(db, user.id, "review.create", f"session:{payload.session_id}")
    db.commit()
    db.refresh(review)
    return review


def _session_export_payload(db: Session, session: ExamSession) -> dict:
    detail = _session_detail(db, session)
    violations = (
        db.query(Violation)
        .filter(Violation.session_id == session.id)
        .order_by(Violation.created_at.asc())
        .all()
    )
    reviews = (
        db.query(Review)
        .filter(Review.session_id == session.id)
        .order_by(Review.created_at.asc())
        .all()
    )
    summary = session.summary or {}
    return {
        "session": detail.model_dump(mode="json"),
        "outcome": summary.get("outcome"),
        "outcome_comment": summary.get("outcome_comment", ""),
        "reviewed_at": summary.get("reviewed_at"),
        "violations": [ViolationResponse.model_validate(item).model_dump(mode="json") for item in violations],
        "reviews": [ReviewResponse.model_validate(item).model_dump(mode="json") for item in reviews],
    }


@router.get("/sessions/{session_id}/export")
def export_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "admin", "proctor")),
    export_format: str = Query("csv", alias="format"),
):
    session = db.get(ExamSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    payload = _session_export_payload(db, session)
    if export_format == "json":
        return payload
    if export_format == "html":
        student = html.escape(payload["session"].get("student_name") or payload["session"].get("student_email") or session_id)
        exam = html.escape(payload["session"].get("exam_title") or "")
        outcome = html.escape(str(payload.get("outcome") or "не вынесено"))
        comment = html.escape(str(payload.get("outcome_comment") or ""))
        rows = []
        for item in payload["violations"]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('type') or ''))}</td>"
                f"<td>{html.escape(str(item.get('message') or ''))}</td>"
                f"<td>{html.escape(str(item.get('severity') or ''))}</td>"
                f"<td>{html.escape(str(item.get('created_at') or ''))}</td>"
                f"<td>{'да' if item.get('is_resolved') else 'нет'}</td>"
                "</tr>"
            )
        body = "".join(rows) or "<tr><td colspan='5'>Событий нет</td></tr>"
        return HTMLResponse(
            f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт IUP</title>
<style>body{{font-family:Segoe UI,sans-serif;padding:24px;color:#111}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;text-align:left}}</style>
</head><body>
<h1>Отчёт по сессии</h1>
<p><strong>{student}</strong> · {exam}</p>
<p>Статус: {html.escape(str(payload["session"].get("status") or ""))} · риск {html.escape(str(payload["session"].get("risk_score") or 0))}</p>
<p>Решение преподавателя: {outcome}</p>
<p>Комментарий: {comment or "—"}</p>
<table><thead><tr><th>Тип</th><th>Сообщение</th><th>Серьёзность</th><th>Время</th><th>Снято</th></tr></thead>
<tbody>{body}</tbody></table>
</body></html>"""
        )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "type", "message", "severity", "created_at", "is_resolved"])
    for item in payload["violations"]:
        writer.writerow([item.get("id"), item.get("type"), item.get("message"), item.get("severity"), item.get("created_at"), item.get("is_resolved")])
    return ExportResponse(session_id=session_id, csv=buffer.getvalue())


@router.post("/webhooks", response_model=WebhookResponse)
def create_webhook(
    payload: WebhookCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    hook = Webhook(url=payload.url, events=payload.events, secret=payload.secret)
    db.add(hook)
    db.commit()
    db.refresh(hook)
    return hook


@router.get("/webhooks", response_model=list[WebhookResponse])
def list_webhooks(db: Session = Depends(get_db), user: User = Depends(require_roles("admin"))):
    return db.query(Webhook).all()


@router.post("/lti/launch", response_model=SessionResponse)
def lti_launch(payload: LTILaunchRequest, request: Request, db: Session = Depends(get_db)):
    if not settings.lti_client_id or not settings.lti_launch_secret:
        raise HTTPException(status_code=404, detail="Not found")
    provided = request.headers.get("X-LTI-Secret", "")
    if not hmac.compare_digest(provided, settings.lti_launch_secret):
        raise HTTPException(status_code=401, detail="Invalid LTI secret")
    exam = db.get(Exam, payload.exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    student, _ = _get_or_create_student(db, payload.student_email.lower(), payload.student_name)
    session = ExamSession(
        exam_id=exam.id,
        student_id=student.id,
        access_token=secrets.token_urlsafe(32),
        status=SessionStatus.PENDING.value,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.post("/admin/backup")
def admin_backup(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    try:
        path = backup_database()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    log_audit(db, user.id, "db.backup", str(path))
    db.commit()
    return {"ok": True, "path": str(path)}


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(
    websocket: WebSocket,
    session_id: str,
    token: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    await websocket.accept()
    bearer = _ws_bearer_token(websocket, token)
    if not bearer:
        try:
            message = await asyncio.wait_for(websocket.receive_json(), timeout=8)
        except Exception:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if not isinstance(message, dict) or message.get("type") != "auth":
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        bearer = str(message.get("token") or "").strip() or None
    try:
        if not bearer:
            raise HTTPException(status_code=401, detail="Missing token")
        user = get_user_from_token(bearer, db)
        if user.role not in STAFF_ROLES:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        session = db.get(ExamSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.attach(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(session_id, websocket)
