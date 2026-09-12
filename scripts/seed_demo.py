#!/usr/bin/env python3
"""Создаёт демо-пользователей, экзамен и сессию. Повторный запуск безопасен."""

from __future__ import annotations

import os
import secrets
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.app.auth import hash_password
from server.app.database import SessionLocal, init_db
from server.app.models import Exam, ExamSession, User
from shared.constants import SessionStatus
from shared.demo import DEMO_EXAM_DESCRIPTION, DEMO_EXAM_TITLE, DEMO_USERS

ACTIVE_STATUSES = {
    SessionStatus.PENDING.value,
    SessionStatus.PRECHECK.value,
    SessionStatus.ACTIVE.value,
}


def _get_or_create_user(db, spec: dict) -> tuple[User, bool]:
    user = db.query(User).filter(User.email == spec["email"]).first()
    if user:
        return user, False
    user = User(
        email=spec["email"],
        password_hash=hash_password(spec["password"]),
        full_name=spec["full_name"],
        role=spec["role"],
    )
    db.add(user)
    db.flush()
    return user, True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    init_db()
    db = SessionLocal()
    try:
        created: list[str] = []
        users: dict[str, User] = {}
        for spec in DEMO_USERS:
            user, was_created = _get_or_create_user(db, spec)
            users[spec["role"]] = user
            if was_created:
                created.append(spec["email"])

        teacher = users["teacher"]
        student = users["student"]

        exam = (
            db.query(Exam)
            .filter(Exam.title == DEMO_EXAM_TITLE, Exam.created_by == teacher.id)
            .order_by(Exam.created_at.desc())
            .first()
        )
        if not exam:
            exam = Exam(
                title=DEMO_EXAM_TITLE,
                description=DEMO_EXAM_DESCRIPTION,
                created_by=teacher.id,
                settings={"open_enrollment": True},
            )
            db.add(exam)
            db.flush()
            created.append(f"exam:{exam.id}")

        session = (
            db.query(ExamSession)
            .filter(
                ExamSession.exam_id == exam.id,
                ExamSession.student_id == student.id,
                ExamSession.status.in_(ACTIVE_STATUSES),
            )
            .order_by(ExamSession.created_at.desc())
            .first()
        )
        if not session:
            session = ExamSession(
                exam_id=exam.id,
                student_id=student.id,
                access_token=secrets.token_urlsafe(32),
                status=SessionStatus.PENDING.value,
            )
            db.add(session)
            db.flush()
            created.append(f"session:{session.id}")

        db.commit()

        print("Demo ready.")
        if created:
            print("Created:", ", ".join(created))
        else:
            print("All demo records already existed.")
        print()
        print("Accounts:")
        for spec in DEMO_USERS:
            print(f"  {spec['role']:8} {spec['email']} / {spec['password']}")
        print()
        print(f"  exam_id     = {exam.id}")
        print(f"  session_id  = {session.id}")
        print(f"  token       = {session.access_token}")
        print()
        print("Teacher dashboard:  http://localhost:8000")
        print("Student cabinet:    http://localhost:8000/student")
        print()
        print("Run agent:")
        print(f"  python -m agent.main --token {session.access_token}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
