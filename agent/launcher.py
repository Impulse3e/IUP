#!/usr/bin/env python3
"""GUI-лаунчер для участника экзамена — Windows, Linux, macOS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from agent.paths import (
    agent_executable,
    agent_python,
    config_path,
    install_path,
    is_frozen,
    project_root,
    venv_pythonw,
)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as error:
    if sys.platform == "win32":
        print("Установите Python с компонентом Tcl/Tk: https://www.python.org/downloads/")
        print("Или откройте кабинет участника в браузере.")
        webbrowser.open("http://localhost:8000/student")
        raise SystemExit(1) from error
    zenity = Path(__file__).resolve().parents[1] / "scripts" / "run_student_zenity.sh"
    if zenity.exists():
        raise SystemExit(subprocess.call([str(zenity)])) from error
    print("Установите: sudo pacman -S tk zenity")
    print("Или откройте http://localhost:8000/student")
    raise SystemExit(1) from error

import httpx

ROOT = project_root()
CONFIG_PATH = config_path()


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"server_url": "http://localhost:8000", "email": "", "password": ""}


def save_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def launch_proctoring(
    token: str,
    server: str,
    parent: tk.Tk | None = None,
    on_complete=None,
) -> None:
    if is_frozen():
        if parent:
            parent.withdraw()
        old_argv = sys.argv.copy()
        sys.argv = [old_argv[0], "--token", token, "--server", server]
        try:
            from agent.main import main as run_agent

            run_agent()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
            if parent:
                parent.deiconify()
            if on_complete:
                on_complete()
        return

    roots = [ROOT, install_path()]
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)

        agent_exe = agent_executable(root)
        if agent_exe:
            subprocess.Popen(
                [str(agent_exe), "--token", token, "--server", server],
                cwd=str(root),
            )
            return

        python = venv_pythonw(root) or agent_python(root)
        agent_dir = root / "agent"
        if agent_dir.is_dir() or (root / "agent" / "main.py").exists():
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.Popen(
                [str(python), "-m", "agent.main", "--token", token, "--server", server],
                cwd=str(root),
                env=env,
                creationflags=creationflags,
            )
            return

    raise FileNotFoundError(
        "Не найден агент прокторинга.\n"
        "Установите IUP (scripts\\install_windows.ps1) или положите IUP-Agent.exe рядом с приложением."
    )


class StudentLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IUP — Участник экзамена")
        self.geometry("560x520")
        self.resizable(False, False)
        if sys.platform == "win32":
            try:
                self.iconbitmap(default="")
            except tk.TclError:
                pass
        self.config = load_config()
        self.token = ""
        self.sessions: list[dict] = []
        self.available: list[dict] = []
        self._build_login()
        if self.config.get("email") and self.config.get("password"):
            self.after(300, self._try_auto_login)

    def _build_login(self) -> None:
        for widget in self.winfo_children():
            widget.destroy()
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Вход участника", font=("", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(frame, text="Сервер").pack(anchor=tk.W, pady=(16, 0))
        self.server_var = tk.StringVar(value=self.config.get("server_url", "http://localhost:8000"))
        ttk.Entry(frame, textvariable=self.server_var, width=50).pack(fill=tk.X)

        ttk.Label(frame, text="Email").pack(anchor=tk.W, pady=(12, 0))
        self.email_var = tk.StringVar(value=self.config.get("email", "student@iup.local"))
        ttk.Entry(frame, textvariable=self.email_var, width=50).pack(fill=tk.X)

        ttk.Label(frame, text="Пароль").pack(anchor=tk.W, pady=(12, 0))
        self.password_var = tk.StringVar(value=self.config.get("password", "student123"))
        ttk.Entry(frame, textvariable=self.password_var, show="*", width=50).pack(fill=tk.X)

        ttk.Button(frame, text="Войти", command=self._login).pack(pady=20, fill=tk.X)
        ttk.Label(
            frame,
            text="Пароль по умолчанию для новых студентов: student123",
            foreground="gray",
        ).pack(anchor=tk.W)

    def _build_sessions(self) -> None:
        for widget in self.winfo_children():
            widget.destroy()
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"Здравствуйте, {self.config.get('email', '')}", font=("", 12)).pack(anchor=tk.W)
        ttk.Label(frame, text="Ваши экзамены", font=("", 16, "bold")).pack(anchor=tk.W, pady=(8, 12))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.session_list = tk.Listbox(list_frame, height=8, font=("", 11))
        self.session_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(list_frame, command=self.session_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.session_list.config(yscrollcommand=scroll.set)

        for session in self.sessions:
            label = f"{session.get('exam_title', 'Экзамен')} — {session.get('status', '')}"
            self.session_list.insert(tk.END, label)

        if self.available:
            ttk.Label(frame, text="Доступны для записи", font=("", 12, "bold")).pack(anchor=tk.W, pady=(12, 4))
            avail_frame = ttk.Frame(frame)
            avail_frame.pack(fill=tk.BOTH, expand=True)
            self.available_list = tk.Listbox(avail_frame, height=4, font=("", 11))
            self.available_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            avail_scroll = ttk.Scrollbar(avail_frame, command=self.available_list.yview)
            avail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.available_list.config(yscrollcommand=avail_scroll.set)
            for exam in self.available:
                self.available_list.insert(tk.END, exam.get("title", "Экзамен"))
        else:
            self.available_list = None

        if self.sessions:
            self.session_list.selection_set(0)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btn_frame, text="Начать экзамен", command=self._start_exam).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        if self.available_list is not None:
            ttk.Button(btn_frame, text="Записаться", command=self._join_exam).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Обновить", command=self._load_sessions).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Выйти", command=self._build_login).pack(side=tk.LEFT)

        ttk.Label(
            frame,
            text="После нажатия «Начать экзамен» откроется окно прокторинга.\nДля выхода из экзамена нажмите Q в окне камеры.",
            foreground="gray",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(12, 0))

    def _try_auto_login(self) -> None:
        try:
            self._login(silent=True)
        except Exception:
            pass

    def _login(self, silent: bool = False) -> None:
        server = self.server_var.get().rstrip("/")
        email = self.email_var.get().strip()
        password = self.password_var.get()
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    f"{server}/api/auth/login",
                    data={"username": email, "password": password},
                )
                response.raise_for_status()
                self.token = response.json()["access_token"]
        except Exception as error:
            if not silent:
                messagebox.showerror("Ошибка входа", str(error))
            return

        self.config.update({"server_url": server, "email": email, "password": password})
        save_config(self.config)
        self._load_sessions()

    def _load_sessions(self) -> None:
        server = self.config["server_url"].rstrip("/")
        try:
            with httpx.Client(timeout=15.0) as client:
                headers = {"Authorization": f"Bearer {self.token}"}
                sessions_resp = client.get(f"{server}/api/my/sessions", headers=headers)
                sessions_resp.raise_for_status()
                self.sessions = sessions_resp.json()
                available_resp = client.get(f"{server}/api/my/available-exams", headers=headers)
                available_resp.raise_for_status()
                self.available = available_resp.json()
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))
            return

        if not self.sessions and not self.available:
            messagebox.showinfo(
                "Нет экзаменов",
                "Открытых экзаменов нет. Попросите преподавателя назначить вам экзамен.",
            )
            return

        self._build_sessions()

    def _join_exam(self) -> None:
        if not self.available_list:
            return
        selection = self.available_list.curselection()
        if not selection:
            messagebox.showwarning("Выбор", "Выберите экзамен для записи.")
            return
        exam = self.available[selection[0]]
        server = self.config["server_url"].rstrip("/")
        try:
            with httpx.Client(timeout=15.0) as client:
                client.post(
                    f"{server}/api/exams/{exam['id']}/join",
                    headers={"Authorization": f"Bearer {self.token}"},
                ).raise_for_status()
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))
            return
        self._load_sessions()

    def _start_exam(self) -> None:
        selection = self.session_list.curselection()
        if not selection:
            messagebox.showwarning("Выбор", "Выберите экзамен из списка.")
            return
        session = self.sessions[selection[0]]
        if session.get("status") == "completed":
            if not messagebox.askyesno("Сессия завершена", "Эта попытка уже завершена. Всё равно продолжить?"):
                return

        token = session["access_token"]
        server = self.config["server_url"].rstrip("/")
        try:
            frozen = is_frozen()
            launch_proctoring(
                token,
                server,
                parent=self if frozen else None,
                on_complete=self._load_sessions if frozen else None,
            )
            if not frozen:
                messagebox.showinfo(
                    "Запущено",
                    f"Прокторинг для «{session.get('exam_title', 'экзамена')}» запущен.\n"
                    "Следуйте инструкциям в окне камеры.",
                )
        except Exception as error:
            messagebox.showerror("Ошибка запуска", str(error))


def main() -> None:
    app = StudentLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
