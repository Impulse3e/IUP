#!/usr/bin/env python3
"""Создаёт демо-экзамен, студента и сессию."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.app.auth import hash_password
from server.app.database import SessionLocal, init_db
from server.app.models import Exam, ExamSession, User
import secrets


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        student = db.query(User).filter(User.email == "student@example.com").first()
        if not student:
            student = User(
                email="student@example.com",
                password_hash=hash_password("student123"),
                full_name="Demo Student",
                role="student",
            )
            db.add(student)
            db.flush()

        teacher = db.query(User).filter(User.email == "teacher@example.com").first()
        if not teacher:
            teacher = User(
                email="teacher@example.com",
                password_hash=hash_password("teacher123"),
                full_name="Demo Teacher",
                role="teacher",
            )
            db.add(teacher)
            db.flush()

        exam = Exam(title="Демо-экзамен", description="Тестовый запуск IUP", created_by=teacher.id)
        db.add(exam)
        db.flush()

        session = ExamSession(
            exam_id=exam.id,
            student_id=student.id,
            access_token=secrets.token_urlsafe(32),
            status="pending",
        )
        db.add(session)
        db.commit()
        print("Demo ready:")
        print(f"  exam_id={exam.id}")
        print(f"  session_id={session.id}")
        print(f"  token={session.access_token}")
        print(f"\nRun agent:\n  python -m agent.main --token {session.access_token}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
