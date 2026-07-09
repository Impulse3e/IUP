#!/usr/bin/env python3
"""Лаунчер участника через zenity (без tkinter)."""

import os
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _pick_session(sessions: list[dict]) -> dict | None:
    lines = [f"{s.get('exam_title', 'Экзамен')}|{s.get('status', '')}|{i}" for i, s in enumerate(sessions)]
    selected = subprocess.run(
        [
            "zenity", "--list", "--title=Выберите экзамен",
            "--column=Экзамен", "--column=Статус", "--column=idx", "--hide-column=3",
        ],
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )
    if selected.returncode != 0:
        return None
    idx = int(selected.stdout.strip().split("|")[-1])
    return sessions[idx]


def _pick_available(available: list[dict]) -> dict | None:
    lines = [f"{e.get('title', 'Экзамен')}|{e.get('description', '')}|{i}" for i, e in enumerate(available)]
    selected = subprocess.run(
        [
            "zenity", "--list", "--title=Записаться на экзамен",
            "--column=Экзамен", "--column=Описание", "--column=idx", "--hide-column=3",
        ],
        input="\n".join(lines),
        capture_output=True,
        text=True,
    )
    if selected.returncode != 0:
        return None
    idx = int(selected.stdout.strip().split("|")[-1])
    return available[idx]


def main() -> None:
    if subprocess.run(["which", "zenity"], capture_output=True).returncode != 0:
        print("Установите zenity: sudo pacman -S zenity")
        sys.exit(1)

    form = subprocess.run(
        [
            "zenity", "--forms", "--title=IUP Student", "--text=Вход участника",
            "--add-entry=Сервер", "--add-entry=Email", "--add-password=Пароль",
            "--separator=|",
        ],
        capture_output=True,
        text=True,
    )
    if form.returncode != 0:
        return
    server, email, password = (form.stdout.strip() + "||").split("|")[:3]
    server = server or os.getenv("IUP_SERVER_URL", "http://localhost:8000")
    server = server.rstrip("/")

    try:
        with httpx.Client(timeout=15.0) as client:
            token = client.post(
                f"{server}/api/auth/login",
                data={"username": email, "password": password},
            ).json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            sessions = client.get(f"{server}/api/my/sessions", headers=headers).json()
            available = client.get(f"{server}/api/my/available-exams", headers=headers).json()
    except Exception as error:
        subprocess.run(["zenity", "--error", f"--text={error}"])
        return

    if not sessions and not available:
        subprocess.run(["zenity", "--info", "--text=Открытых экзаменов нет. Попросите преподавателя назначить вам экзамен."])
        return

    session = None
    if sessions:
        session = _pick_session(sessions)
    elif available:
        exam = _pick_available(available)
        if not exam:
            return
        try:
            with httpx.Client(timeout=15.0) as client:
                session = client.post(
                    f"{server}/api/exams/{exam['id']}/join",
                    headers={"Authorization": f"Bearer {token}"},
                ).json()
        except Exception as error:
            subprocess.run(["zenity", "--error", f"--text={error}"])
            return

    if not session:
        return

    python = ROOT / ".venv" / "bin" / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.Popen(
        [str(python), "-m", "agent.main", "--token", session["access_token"], "--server", server],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run([
        "zenity", "--info",
        f"--text=Прокторинг запущен: {session.get('exam_title', 'экзамен')}\n\n"
        "Следуйте инструкциям в окне камеры.\nДля выхода нажмите Q.",
    ])


if __name__ == "__main__":
    main()
