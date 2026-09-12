#!/usr/bin/env python3
"""GUI-лаунчер для участника экзамена — Windows, Linux, macOS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from agent.http_util import http_timeout, normalize_server_url
from agent.ui import configure_stdio

configure_stdio()
from agent.paths import (
    agent_executable,
    agent_python,
    config_path,
    install_path,
    is_frozen,
    project_root,
    venv_python,
    writable_root,
)
from agent.security.consent import ask_consent

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

BG = "#09111f"
PANEL = "#131c30"
SURFACE = "#0e1626"
TEXT = "#f3f6ff"
MUTED = "#8ea0b8"
ACCENT = "#4c8dff"
ACCENT2 = "#38d6f2"
BORDER = "#27344d"
OK = "#34d399"
WARN = "#fbbf24"
DANGER = "#fb7185"
FONT = ("Segoe UI", 11)
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_H2 = ("Segoe UI", 16, "bold")
FONT_SMALL = ("Segoe UI", 10)
FONT_BADGE = ("Segoe UI", 9, "bold")

STATUS_LABELS = {
    "pending": "Ожидает",
    "precheck": "Проверка",
    "active": "Идёт",
    "completed": "Завершён",
    "compromised": "Нарушения",
    "cancelled": "Отменён",
}

STATUS_COLORS = {
    "pending": WARN,
    "precheck": WARN,
    "active": OK,
    "completed": MUTED,
    "compromised": DANGER,
    "cancelled": MUTED,
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status or "—")


def prepare_dpi() -> None:
    """Ask Windows for per-monitor DPI so the UI follows display scaling."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _login_error_text(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    refused = (
        "10061" in text
        or "connection refused" in lowered
        or "actively refused" in lowered
        or "connecterror" in type(error).__name__.lower()
        or isinstance(error, httpx.ConnectError)
    )
    if refused:
        return (
            "Сервер IUP не запущен (ошибка подключения).\n\n"
            "Запустите scripts\\run_server.bat и подождите строку "
            "«Uvicorn running on http://0.0.0.0:8000», затем войдите снова."
        )
    if isinstance(error, (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TimeoutException)):
        return (
            "Сервер IUP не отвечает.\n\n"
            "Проверьте, что окно сервера открыто, затем повторите вход."
        )
    return text


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
        sys.argv = [old_argv[0], "--token", token, "--server", server, "--consent-accepted"]
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
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            subprocess.Popen(
                [str(agent_exe), "--token", token, "--server", server, "--consent-accepted"],
                cwd=str(root),
                env=env,
            )
            return

        python = venv_python(root) or agent_python(root)
        agent_dir = root / "agent"
        if agent_dir.is_dir() or (root / "agent" / "main.py").exists():
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            log_path = writable_root() / "agent.log"
            log_handle = open(log_path, "ab")
            subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "agent.main",
                    "--token",
                    token,
                    "--server",
                    server,
                    "--consent-accepted",
                ],
                cwd=str(root),
                env=env,
                stdout=log_handle,
                stderr=log_handle,
                stdin=subprocess.DEVNULL,
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
        self.configure(bg=BG)
        self._dip = 1.0
        self._wrap_labels: list[tk.Label] = []
        self._canvas = None
        self._canvas_window = None
        self._init_scaling()
        if sys.platform == "win32":
            try:
                self.iconbitmap(default="")
            except tk.TclError:
                pass
        self._apply_theme()
        self.settings = load_config()
        self.token = ""
        self.sessions: list[dict] = []
        self.available: list[dict] = []
        self._busy = False
        self._request_id = 0
        self._selected_session = 0
        self._selected_available = 0
        self.status_var = tk.StringVar(value="")
        self._action_buttons: list[tk.Widget] = []
        self.login_btn = None
        self.bind("<Configure>", self._on_resize)
        self._build_login()
        if self.settings.get("email") and self.settings.get("password"):
            self.after_idle(self._try_auto_login)

    def _scaled(self, value: int) -> int:
        return max(1, int(round(value * self._dip)))

    def _init_scaling(self) -> None:
        self.update_idletasks()
        dpi = float(self.winfo_fpixels("1i") or 96.0)
        self._dip = max(1.0, dpi / 96.0)
        try:
            self.tk.call("tk", "scaling", dpi / 72.0)
        except tk.TclError:
            pass
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = min(self._scaled(720), max(self._scaled(480), int(screen_w * 0.5)))
        height = min(self._scaled(820), max(self._scaled(560), int(screen_h * 0.75)))
        self.geometry(f"{width}x{height}")
        self.minsize(self._scaled(380), self._scaled(420))
        self.resizable(True, True)

    def _on_resize(self, event) -> None:
        if event.widget is not self:
            return
        width = int(getattr(event, "width", 0) or self.winfo_width())
        if width < 80:
            return
        wrap = max(self._scaled(180), width - self._scaled(64))
        for label in list(self._wrap_labels):
            try:
                if label.winfo_exists():
                    label.configure(wraplength=wrap)
            except tk.TclError:
                pass

    def _remember_wrap(self, label: tk.Label) -> tk.Label:
        self._wrap_labels.append(label)
        return label

    def _scroll_area(self, parent) -> tk.Frame:
        holder = tk.Frame(parent, bg=BG)
        holder.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(holder, bg=BG, highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=BG)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def on_inner_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event) -> None:
            if event.widget is not canvas or event.width < 80:
                return
            canvas.itemconfigure(window_id, width=event.width)
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def wheel(event) -> str:
            delta = getattr(event, "delta", 0) or 0
            if delta == 0 and getattr(event, "num", None) in {4, 5}:
                delta = 120 if event.num == 4 else -120
            canvas.yview_scroll(int(-delta / 120), "units")
            return "break"

        self.bind_all("<MouseWheel>", wheel)
        self.bind_all("<Button-4>", wheel)
        self.bind_all("<Button-5>", wheel)

        self._canvas = canvas
        self._canvas_window = window_id
        return inner

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=FONT)
        style.configure("TCheckbutton", background=BG, foreground=MUTED, font=FONT_SMALL)
        style.map("TCheckbutton", background=[("active", BG)], foreground=[("active", TEXT)])
        style.configure("Vertical.TScrollbar", background=PANEL, troughcolor=BG, bordercolor=BORDER)

    def _clear(self) -> None:
        self._action_buttons = []
        self.login_btn = None
        self._wrap_labels = []
        self._canvas = None
        self._canvas_window = None
        try:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")
        except tk.TclError:
            pass
        for widget in self.winfo_children():
            widget.destroy()

    def _entry(self, parent, variable: tk.StringVar, show: str = "") -> tk.Entry:
        entry = tk.Entry(
            parent,
            textvariable=variable,
            show=show,
            bg=SURFACE,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            font=FONT,
        )
        entry.pack(fill=tk.X, ipady=self._scaled(8), pady=(self._scaled(4), 0))
        return entry

    def _label(self, parent, text: str, **kwargs) -> tk.Label:
        label = tk.Label(parent, text=text, bg=parent.cget("bg"), fg=MUTED, font=FONT_SMALL, anchor="w", **kwargs)
        label.pack(fill=tk.X, pady=(self._scaled(12), 0))
        return label

    def _button(self, parent, text: str, command, accent: bool = False) -> tk.Button:
        if accent:
            bg, fg, active = ACCENT, "#ffffff", "#3b7af0"
        else:
            bg, fg, active = PANEL, TEXT, "#1b263c"
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active,
            activeforeground=fg,
            relief="flat",
            font=("Segoe UI", 11, "bold") if accent else FONT,
            cursor="hand2",
            padx=self._scaled(14),
            pady=self._scaled(8),
            highlightthickness=0,
            bd=0,
        )
        return button

    def _brand(self, parent, subtitle: str) -> None:
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill=tk.X, pady=(0, self._scaled(18)))
        mark = tk.Label(
            row,
            text="IUP",
            bg=ACCENT2,
            fg="#061018",
            font=("Segoe UI", 10, "bold"),
            width=4,
            pady=self._scaled(8),
        )
        mark.pack(side=tk.LEFT)
        copy = tk.Frame(row, bg=parent.cget("bg"))
        copy.pack(side=tk.LEFT, padx=self._scaled(12), fill=tk.X, expand=True)
        tk.Label(copy, text="IUP Proctoring", bg=parent.cget("bg"), fg=TEXT, font=("Segoe UI", 13, "bold"), anchor="w").pack(fill=tk.X)
        self._remember_wrap(
            tk.Label(copy, text=subtitle, bg=parent.cget("bg"), fg=MUTED, font=FONT_SMALL, anchor="w", justify="left")
        ).pack(fill=tk.X)

    def _card(self, parent, selected: bool = False) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=PANEL,
            highlightthickness=1,
            highlightbackground=ACCENT if selected else BORDER,
            padx=self._scaled(14),
            pady=self._scaled(12),
        )
        card.pack(fill=tk.X, pady=self._scaled(6))
        return card

    def _badge(self, parent, status: str) -> None:
        color = STATUS_COLORS.get(status, MUTED)
        tk.Label(
            parent,
            text=status_label(status),
            bg=parent.cget("bg"),
            fg=color,
            font=FONT_BADGE,
        ).pack(anchor="w")

    def _build_login(self) -> None:
        self._request_id += 1
        self._set_busy(False)
        self._clear()
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill=tk.BOTH, expand=True, padx=self._scaled(28), pady=self._scaled(24))

        self._brand(wrap, "Кабинет участника")
        tk.Label(wrap, text="Вход", bg=BG, fg=TEXT, font=FONT_TITLE, anchor="w").pack(fill=tk.X)
        self._remember_wrap(
            tk.Label(
                wrap,
                text="Войдите, чтобы увидеть экзамены и запустить прокторинг.",
                bg=BG,
                fg=MUTED,
                font=FONT,
                justify="left",
                anchor="w",
            )
        ).pack(fill=tk.X, pady=(self._scaled(6), self._scaled(8)))

        self.show_server = tk.BooleanVar(
            value="localhost" not in str(self.settings.get("server_url", "")).lower()
            and "127.0.0.1" not in str(self.settings.get("server_url", ""))
        )
        self.server_var = tk.StringVar(value=self.settings.get("server_url", "http://localhost:8000"))
        self.email_var = tk.StringVar(value=self.settings.get("email", "student@iup.local"))
        self.password_var = tk.StringVar(value=self.settings.get("password", "student123"))

        self._label(wrap, "Email")
        email_entry = self._entry(wrap, self.email_var)
        self._label(wrap, "Пароль")
        password_entry = self._entry(wrap, self.password_var, show="*")
        email_entry.bind("<Return>", lambda _event: self._login())
        password_entry.bind("<Return>", lambda _event: self._login())

        self.server_box = tk.Frame(wrap, bg=BG)
        self._label(self.server_box, "Адрес сервера")
        self._entry(self.server_box, self.server_var)
        tk.Label(
            self.server_box,
            text="С другого компьютера укажите http://IP-сервера:8000",
            bg=BG,
            fg=MUTED,
            font=FONT_SMALL,
            anchor="w",
            justify="left",
        ).pack(fill=tk.X, pady=(self._scaled(4), 0))

        def toggle_server() -> None:
            if self.show_server.get():
                self.server_box.pack(fill=tk.X, before=self.login_btn)
            else:
                self.server_box.pack_forget()

        tk.Checkbutton(
            wrap,
            text="Показать адрес сервера",
            variable=self.show_server,
            command=toggle_server,
            bg=BG,
            fg=MUTED,
            selectcolor=SURFACE,
            activebackground=BG,
            activeforeground=TEXT,
            font=FONT_SMALL,
            highlightthickness=0,
            bd=0,
            anchor="w",
        ).pack(fill=tk.X, pady=(self._scaled(14), 0))

        self.login_btn = self._button(wrap, "Войти", self._login, accent=True)
        self.login_btn.pack(fill=tk.X, pady=(self._scaled(18), self._scaled(8)))
        self._action_buttons = [self.login_btn]
        if self.show_server.get():
            self.server_box.pack(fill=tk.X, before=self.login_btn)

        tk.Label(wrap, textvariable=self.status_var, bg=BG, fg=ACCENT2, font=FONT_SMALL, anchor="w", justify="left").pack(fill=tk.X)
        self._remember_wrap(
            tk.Label(
                wrap,
                text="Демо: student@iup.local / student123",
                bg=BG,
                fg=MUTED,
                font=FONT_SMALL,
                anchor="w",
                justify="left",
            )
        ).pack(fill=tk.X, pady=(self._scaled(8), 0))

    def _build_sessions(self) -> None:
        self._clear()
        shell = tk.Frame(self, bg=BG)
        shell.pack(fill=tk.BOTH, expand=True)

        head = tk.Frame(shell, bg=BG)
        head.pack(fill=tk.X, padx=self._scaled(24), pady=(self._scaled(18), self._scaled(8)))
        self._brand(head, self.settings.get("email", ""))
        tk.Label(head, text="Ваши экзамены", bg=BG, fg=TEXT, font=FONT_H2, anchor="w").pack(fill=tk.X)

        body = tk.Frame(shell, bg=BG, padx=self._scaled(24))
        body.pack(fill=tk.BOTH, expand=True)
        wrap = self._scroll_area(body)

        if not self.sessions:
            empty = self._card(wrap)
            self._remember_wrap(
                tk.Label(
                    empty,
                    text="Назначенных экзаменов пока нет. Попросите преподавателя открыть запись или назначить попытку.",
                    bg=PANEL,
                    fg=MUTED,
                    font=FONT,
                    justify="left",
                    anchor="w",
                )
            ).pack(fill=tk.X)
            self.session_cards = []
        else:
            self.session_cards = []
            if self._selected_session >= len(self.sessions):
                self._selected_session = 0
            for index, session in enumerate(self.sessions):
                selected = index == self._selected_session
                card = self._card(wrap, selected=selected)
                self.session_cards.append(card)
                title = self._remember_wrap(
                    tk.Label(
                        card,
                        text=session.get("exam_title") or "Экзамен",
                        bg=PANEL,
                        fg=TEXT,
                        font=("Segoe UI", 13, "bold"),
                        anchor="w",
                        justify="left",
                    )
                )
                title.pack(fill=tk.X)
                self._badge(card, str(session.get("status") or ""))
                meta = tk.Label(
                    card,
                    text=f"Риск: {float(session.get('risk_score') or 0):.1f}  ·  нажмите, чтобы выбрать",
                    bg=PANEL,
                    fg=MUTED,
                    font=FONT_SMALL,
                    anchor="w",
                    justify="left",
                )
                meta.pack(fill=tk.X, pady=(self._scaled(6), 0))
                self._remember_wrap(meta)
                for widget in (card, *card.winfo_children()):
                    widget.bind("<Button-1>", lambda _event, i=index: self._select_session(i))
                    widget.bind("<Double-Button-1>", lambda _event, i=index: self._select_and_start(i))
                    widget.bind("<MouseWheel>", self._forward_wheel)

        if self.available:
            tk.Label(wrap, text="Доступны для записи", bg=BG, fg=TEXT, font=("Segoe UI", 13, "bold"), anchor="w").pack(
                fill=tk.X, pady=(self._scaled(16), self._scaled(4))
            )
            self.available_cards = []
            if self._selected_available >= len(self.available):
                self._selected_available = 0
            for index, exam in enumerate(self.available):
                selected = index == self._selected_available
                card = self._card(wrap, selected=selected)
                self.available_cards.append(card)
                tk.Label(
                    card,
                    text=exam.get("title") or "Экзамен",
                    bg=PANEL,
                    fg=TEXT,
                    font=("Segoe UI", 12, "bold"),
                    anchor="w",
                    justify="left",
                ).pack(fill=tk.X)
                self._remember_wrap(
                    tk.Label(
                        card,
                        text=exam.get("description") or "Без описания",
                        bg=PANEL,
                        fg=MUTED,
                        font=FONT_SMALL,
                        anchor="w",
                        justify="left",
                    )
                ).pack(fill=tk.X, pady=(self._scaled(4), 0))
                for widget in (card, *card.winfo_children()):
                    widget.bind("<Button-1>", lambda _event, i=index: self._select_available(i))
                    widget.bind("<MouseWheel>", self._forward_wheel)
        else:
            self.available_cards = []

        foot = tk.Frame(shell, bg=BG)
        foot.pack(fill=tk.X, padx=self._scaled(24), pady=(self._scaled(8), self._scaled(16)))
        btns = tk.Frame(foot, bg=BG)
        btns.pack(fill=tk.X)
        for column in range(2):
            btns.grid_columnconfigure(column, weight=1, uniform="btn")
        start_btn = self._button(btns, "Начать экзамен", self._start_exam, accent=True)
        start_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, self._scaled(8)))
        self._action_buttons.append(start_btn)
        column = 0
        row = 1
        extra = []
        if self.available:
            extra.append(("Записаться", self._join_exam))
        extra.extend([("Обновить", self._load_sessions), ("Выйти", self._build_login)])
        for text, command in extra:
            button = self._button(btns, text, command)
            button.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else self._scaled(8), 0), pady=(0, self._scaled(6)))
            self._action_buttons.append(button)
            column += 1
            if column > 1:
                column = 0
                row += 1

        tk.Label(foot, textvariable=self.status_var, bg=BG, fg=ACCENT2, font=FONT_SMALL, anchor="w", justify="left").pack(fill=tk.X, pady=(self._scaled(8), 0))
        self._remember_wrap(
            tk.Label(
                foot,
                text="После запуска откроется окно камеры «IUP Proctoring». Смотрите в камеру. Выход — клавиша Q.",
                bg=BG,
                fg=MUTED,
                font=FONT_SMALL,
                justify="left",
                anchor="w",
            )
        ).pack(fill=tk.X, pady=(self._scaled(8), 0))

    def _forward_wheel(self, event):
        canvas = self._canvas
        if canvas is None:
            return
        delta = getattr(event, "delta", 0) or 0
        canvas.yview_scroll(int(-delta / 120), "units")
        return "break"

    def _paint_selection(self, cards: list[tk.Frame], selected: int) -> None:
        for index, card in enumerate(cards):
            try:
                card.configure(highlightbackground=ACCENT if index == selected else BORDER)
            except tk.TclError:
                pass

    def _select_session(self, index: int) -> None:
        self._selected_session = index
        self._paint_selection(self.session_cards, index)

    def _select_available(self, index: int) -> None:
        self._selected_available = index
        self._paint_selection(self.available_cards, index)

    def _select_and_start(self, index: int) -> None:
        self._selected_session = index
        self._paint_selection(self.session_cards, index)
        self._start_exam()

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self.status_var.set(status)
        try:
            self.configure(cursor="watch" if busy else "")
        except tk.TclError:
            pass
        for button in list(self._action_buttons):
            try:
                button.configure(state="disabled" if busy else "normal")
            except tk.TclError:
                pass
        login_btn = getattr(self, "login_btn", None)
        if login_btn is not None:
            try:
                login_btn.configure(state="disabled" if busy else "normal")
            except tk.TclError:
                pass

    def _in_background(self, work, on_success, on_error=None, status: str = "") -> None:
        if self._busy:
            return
        self._request_id += 1
        req_id = self._request_id
        self._set_busy(True, status)
        self.update_idletasks()

        def runner() -> None:
            try:
                result = work()
            except Exception as error:
                self.after(0, lambda e=error, i=req_id: self._bg_fail(e, on_error, i))
                return
            self.after(0, lambda r=result, i=req_id: self._bg_ok(r, on_success, i))

        threading.Thread(target=runner, daemon=True).start()

    def _bg_ok(self, result, on_success, req_id: int) -> None:
        if req_id != self._request_id:
            return
        self._set_busy(False)
        if not self.winfo_exists():
            return
        try:
            on_success(result)
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _bg_fail(self, error: Exception, on_error, req_id: int) -> None:
        if req_id != self._request_id:
            return
        self._set_busy(False)
        if not self.winfo_exists():
            return
        if on_error:
            on_error(error)

    def _try_auto_login(self) -> None:
        self._login(silent=True)

    def _login(self, silent: bool = False) -> None:
        server = self.server_var.get().rstrip("/")
        email = self.email_var.get().strip()
        password = self.password_var.get()

        def work() -> dict:
            base = normalize_server_url(server)
            with httpx.Client(timeout=http_timeout()) as client:
                response = client.post(
                    f"{base}/api/auth/login",
                    data={"username": email, "password": password},
                )
                response.raise_for_status()
                token = response.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}"}
                sessions_resp = client.get(f"{base}/api/my/sessions", headers=headers)
                sessions_resp.raise_for_status()
                available_resp = client.get(f"{base}/api/my/available-exams", headers=headers)
                available_resp.raise_for_status()
                return {
                    "token": token,
                    "sessions": sessions_resp.json(),
                    "available": available_resp.json(),
                }

        def on_success(payload: dict) -> None:
            self.token = payload["token"]
            self.sessions = payload["sessions"]
            self.available = payload["available"]
            self.settings.update({"server_url": server, "email": email, "password": password})
            save_config(self.settings)
            self._show_sessions()

        def on_error(error: Exception) -> None:
            if not silent:
                messagebox.showerror("Ошибка входа", _login_error_text(error))

        self._in_background(work, on_success, on_error, "Подключение к серверу...")

    def _load_sessions(self) -> None:
        server = normalize_server_url(self.settings["server_url"])

        def work() -> dict:
            with httpx.Client(timeout=http_timeout()) as client:
                headers = {"Authorization": f"Bearer {self.token}"}
                sessions_resp = client.get(f"{server}/api/my/sessions", headers=headers)
                sessions_resp.raise_for_status()
                available_resp = client.get(f"{server}/api/my/available-exams", headers=headers)
                available_resp.raise_for_status()
                return {"sessions": sessions_resp.json(), "available": available_resp.json()}

        def on_success(payload: dict) -> None:
            self.sessions = payload["sessions"]
            self.available = payload["available"]
            self._show_sessions()

        def on_error(error: Exception) -> None:
            messagebox.showerror("Ошибка", str(error))

        self._in_background(work, on_success, on_error, "Обновление списка...")

    def _show_sessions(self) -> None:
        try:
            self._build_sessions()
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))
            self._build_login()

    def _join_exam(self) -> None:
        if not self.available:
            return
        exam = self.available[self._selected_available]
        server = normalize_server_url(self.settings["server_url"])

        def work() -> None:
            with httpx.Client(timeout=http_timeout()) as client:
                client.post(
                    f"{server}/api/exams/{exam['id']}/join",
                    headers={"Authorization": f"Bearer {self.token}"},
                ).raise_for_status()

        def on_success(_result) -> None:
            self._load_sessions()

        def on_error(error: Exception) -> None:
            messagebox.showerror("Ошибка", str(error))

        self._in_background(work, on_success, on_error, "Запись на экзамен...")

    def _start_exam(self) -> None:
        if not self.sessions:
            messagebox.showinfo("IUP", "Сначала запишитесь на экзамен или дождитесь назначения.")
            return
        session = self.sessions[self._selected_session]
        if session.get("status") == "completed":
            if not messagebox.askyesno("Сессия завершена", "Эта попытка уже завершена. Всё равно продолжить?"):
                return

        token = session["access_token"]
        server = normalize_server_url(self.settings["server_url"])
        if self._busy:
            return
        if not ask_consent():
            self.status_var.set("Согласие не получено. Экзамен не запущен.")
            return
        try:
            frozen = is_frozen()
            launch_proctoring(
                token,
                server,
                parent=self if frozen else None,
                on_complete=self._load_sessions if frozen else None,
            )
            if not frozen:
                title = session.get("exam_title") or "экзамена"
                self.status_var.set(f"Прокторинг «{title}» запущен. Смотрите в окно камеры, выход — Q.")
        except Exception as error:
            messagebox.showerror("Ошибка запуска", str(error))


def main() -> None:
    prepare_dpi()
    app = StudentLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
