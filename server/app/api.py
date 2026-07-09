import csv
import io
import os
import secrets
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from server.app.auth import create_access_token, get_current_user, hash_password, require_roles, verify_password
from server.app.database import get_db
from server.app.models import Exam, ExamSession, Evidence, Review, User, VideoChunk, Violation, Webhook
from server.app.schemas import (
    ConsentRequest,
    EventRequest,
    ExamCreate,
    ExamResponse,
    ExportResponse,
    HeartbeatRequest,
    IdentityRequest,
    LTILaunchRequest,
    LoginRequest,
    ReviewCreate,
    ReviewResponse,
    SessionCreate,
    SessionCreateByEmail,
    SessionEndRequest,
    SessionResponse,
    SessionWithExamResponse,
    TokenResponse,
    UserCreate,
    UserResponse,
    ViolationResponse,
    WebhookCreate,
    WebhookResponse,
)
from server.app.services import dispatch_webhooks, log_audit, risk_delta
from server.app.storage import get_storage
from server.app.websocket import manager
from shared.constants import SessionStatus

router = APIRouter()


def _session_by_token(db: Session, token: str) -> ExamSession:
    session = db.query(ExamSession).filter(ExamSession.access_token == token).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _session_detail(db: Session, session: ExamSession) -> SessionWithExamResponse:
    exam = db.get(Exam, session.exam_id)
    student = db.get(User, session.student_id)
    data = SessionResponse.model_validate(session).model_dump()
    data["exam_title"] = exam.title if exam else ""
    data["student_name"] = student.full_name if student else ""
    data["student_email"] = student.email if student else ""
    return SessionWithExamResponse(**data)


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


@router.post("/exams", response_model=ExamResponse)
def create_exam(
    payload: ExamCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "admin")),
):
    settings = payload.settings or {}
    settings.setdefault("open_enrollment", True)
    exam = Exam(
        title=payload.title,
        description=payload.description,
        created_by=user.id,
        settings=settings,
        retention_days=payload.retention_days,
    )
    db.add(exam)
    db.flush()
    log_audit(db, user.id, "exam.create", f"exam:{exam.id}")

    if payload.student_email.strip():
        email = payload.student_email.strip().lower()
        student = db.query(User).filter(User.email == email).first()
        if not student:
            student = User(
                email=email,
                password_hash=hash_password("student123"),
                full_name=payload.student_name or email.split("@")[0],
                role="student",
            )
            db.add(student)
            db.flush()
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
    return exam


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


@router.get("/my/sessions", response_model=list[SessionWithExamResponse])
def my_sessions(db: Session = Depends(get_db), user: User = Depends(require_roles("student"))):
    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.student_id == user.id)
        .order_by(ExamSession.created_at.desc())
        .all()
    )
    return [_session_detail(db, session) for session in sessions]


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
        settings = exam.settings or {}
        if settings.get("open_enrollment", True) is False:
            continue
        if exam.id in active_exam_ids:
            continue
        result.append(exam)
    return result


@router.post("/exams/{exam_id}/join", response_model=SessionWithExamResponse)
def join_exam(
    exam_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    exam = db.get(Exam, exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    settings = exam.settings or {}
    if settings.get("open_enrollment", True) is False:
        raise HTTPException(status_code=403, detail="Экзамен доступен только по приглашению")

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
        return _session_detail(db, active)

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
    return _session_detail(db, session)


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


@router.post("/exams/{exam_id}/sessions/by-email", response_model=SessionWithExamResponse)
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
    student = db.query(User).filter(User.email == email).first()
    if not student:
        student = User(
            email=email,
            password_hash=hash_password("student123"),
            full_name=payload.full_name or email.split("@")[0],
            role="student",
        )
        db.add(student)
        db.flush()
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
    return _session_detail(db, session)


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
    install = os.getenv("IUP_INSTALL_PATH", os.path.expanduser("~/IUP"))
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="{install}"
SERVER="{server}"
TOKEN="{session.access_token}"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "Не найден $ROOT/.venv/bin/python"
  echo "Установите IUP или запустите приложение IUP Student"
  read -r _
  exit 1
fi
cd "$ROOT"
export PYTHONPATH=.
exec "$ROOT/.venv/bin/python" -m agent.main --token "$TOKEN" --server "$SERVER"
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
    install = os.getenv("IUP_INSTALL_PATH", os.path.expandvars(r"%USERPROFILE%\IUP"))
    script = f"""@echo off
setlocal EnableExtensions
set "ROOT={install}"
set "SERVER={server}"
set "TOKEN={session.access_token}"
cd /d "%ROOT%"
if not exist "%ROOT%\\.venv\\Scripts\\python.exe" (
  echo Не найден IUP в %ROOT%
  echo Установите IUP или запустите приложение IUP Student
  pause
  exit /b 1
)
set "PYTHONPATH=%ROOT%"
start "" "%ROOT%\\.venv\\Scripts\\pythonw.exe" -m agent.main --token "%TOKEN%" --server "%SERVER%"
"""
    return PlainTextResponse(script, media_type="application/x-bat", headers={
        "Content-Disposition": f'attachment; filename="iup-exam-{session_id[:8]}.bat"'
    })


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    return db.query(ExamSession).order_by(ExamSession.created_at.desc()).limit(200).all()


@router.get("/sessions/token/{token}", response_model=SessionResponse)
def get_session_by_token(token: str, db: Session = Depends(get_db)):
    return _session_by_token(db, token)


@router.post("/sessions/token/{token}/consent")
async def accept_consent(token: str, payload: ConsentRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    if not payload.accepted:
        session.status = SessionStatus.CANCELLED.value
    else:
        session.consent_at = datetime.utcnow()
        session.status = SessionStatus.PRECHECK.value
    db.commit()
    await manager.broadcast(session.id, {"type": "consent", "session_id": session.id})
    return {"status": session.status}


@router.post("/sessions/token/{token}/identity")
async def verify_identity(token: str, payload: IdentityRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    student = db.get(User, session.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student.face_embedding = {"vector": payload.embedding}
    session.identity_verified = True
    db.commit()
    await manager.broadcast(session.id, {"type": "identity_verified", "session_id": session.id})
    return {"identity_verified": True}


@router.post("/sessions/token/{token}/start", response_model=SessionResponse)
async def start_session(token: str, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    if not session.consent_at:
        raise HTTPException(status_code=400, detail="Consent required")
    if not session.identity_verified:
        raise HTTPException(status_code=400, detail="Identity verification required")
    session.status = SessionStatus.ACTIVE.value
    session.started_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    await manager.broadcast_all({"type": "session_started", "session": SessionResponse.model_validate(session).model_dump(mode="json")})
    await dispatch_webhooks(db, "session.started", {"session_id": session.id})
    return session


@router.post("/sessions/token/{token}/heartbeat")
async def heartbeat(token: str, payload: HeartbeatRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
    session.last_heartbeat = datetime.utcnow()
    session.agent_version = payload.agent_version
    if payload.status == "compromised":
        session.status = SessionStatus.COMPROMISED.value
    db.commit()
    await manager.broadcast(session.id, {"type": "heartbeat", "session_id": session.id, "payload": payload.model_dump()})
    return {"ok": True}


@router.post("/sessions/token/{token}/events", response_model=ViolationResponse)
async def post_event(token: str, payload: EventRequest, db: Session = Depends(get_db)):
    session = _session_by_token(db, token)
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
    session.status = SessionStatus.COMPLETED.value
    session.ended_at = datetime.utcnow()
    session.summary = payload.summary
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
):
    session = _session_by_token(db, token)
    content = await file.read()
    relative = f"sessions/{session.id}/evidence/{uuid.uuid4()}_{file.filename}"
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


@router.post("/sessions/token/{token}/chunks")
async def upload_chunk(
    token: str,
    source: Annotated[str, Form()],
    chunk_index: Annotated[int, Form()],
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    session = _session_by_token(db, token)
    content = await file.read()
    relative = f"sessions/{session.id}/chunks/{source}_{chunk_index:06d}.bin"
    path = get_storage().save_bytes(relative, content)
    chunk = VideoChunk(session_id=session.id, source=source, chunk_index=chunk_index, path=path)
    db.add(chunk)
    db.commit()
    return {"path": path}


@router.get("/sessions/{session_id}/violations", response_model=list[ViolationResponse])
def session_violations(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
    return (
        db.query(Violation)
        .filter(Violation.session_id == session_id)
        .order_by(Violation.created_at.asc())
        .all()
    )


@router.post("/reviews", response_model=ReviewResponse)
def create_review(
    payload: ReviewCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("proctor", "teacher", "admin")),
):
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


@router.get("/sessions/{session_id}/export", response_model=ExportResponse)
def export_session(
    session_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("teacher", "admin", "proctor")),
):
    violations = db.query(Violation).filter(Violation.session_id == session_id).all()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "type", "message", "severity", "created_at", "is_resolved"])
    for item in violations:
        writer.writerow([item.id, item.type, item.message, item.severity, item.created_at, item.is_resolved])
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
def lti_launch(payload: LTILaunchRequest, db: Session = Depends(get_db)):
    exam = db.get(Exam, payload.exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    student = db.query(User).filter(User.email == payload.student_email).first()
    if not student:
        student = User(
            email=payload.student_email,
            password_hash=hash_password(secrets.token_urlsafe(16)),
            full_name=payload.student_name,
            role="student",
        )
        db.add(student)
        db.flush()
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


@router.websocket("/ws/sessions/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(session_id, websocket)
