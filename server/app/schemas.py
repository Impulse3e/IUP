from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "student"


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ExamCreate(BaseModel):
    title: str
    description: str = ""
    settings: dict[str, Any] = Field(default_factory=lambda: {"open_enrollment": True})
    retention_days: int = 90
    student_email: str = ""
    student_name: str = ""


class ExamResponse(BaseModel):
    id: str
    title: str
    description: str
    settings: dict[str, Any]
    retention_days: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    student_id: str


class SessionCreateByEmail(BaseModel):
    email: str
    full_name: str = ""


class SessionResponse(BaseModel):
    id: str
    exam_id: str
    student_id: str
    access_token: str
    status: str
    risk_score: float
    consent_at: datetime | None
    identity_verified: bool
    started_at: datetime | None
    ended_at: datetime | None
    last_heartbeat: datetime | None
    agent_version: str
    summary: dict[str, Any] | None

    model_config = {"from_attributes": True}


class SessionWithExamResponse(SessionResponse):
    exam_title: str = ""
    student_name: str = ""
    student_email: str = ""


class EventRequest(BaseModel):
    type: str
    message: str
    severity: str = "warning"
    payload: dict[str, Any] = Field(default_factory=dict)
    is_reminder: bool = False
    is_resolved: bool = False


class HeartbeatRequest(BaseModel):
    agent_version: str = "1.0.0"
    status: str = "active"
    payload: dict[str, Any] = Field(default_factory=dict)


class ConsentRequest(BaseModel):
    accepted: bool
    policy_version: str = "1.0"


class IdentityRequest(BaseModel):
    embedding: list[float]


class SessionEndRequest(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)


class ReviewCreate(BaseModel):
    session_id: str
    violation_id: str | None = None
    decision: str
    comment: str = ""


class ReviewResponse(BaseModel):
    id: str
    session_id: str
    violation_id: str | None
    reviewer_id: str
    decision: str
    comment: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ViolationResponse(BaseModel):
    id: str
    session_id: str
    type: str
    message: str
    severity: str
    payload: dict[str, Any]
    is_reminder: bool
    is_resolved: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class WebhookCreate(BaseModel):
    url: str
    events: list[str] = Field(default_factory=list)
    secret: str = ""


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: list[str]
    active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class LTILaunchRequest(BaseModel):
    exam_id: str
    student_email: EmailStr
    student_name: str


class ExportResponse(BaseModel):
    session_id: str
    csv: str
