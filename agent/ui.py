"""Shared GUI helpers for frozen / windowed builds."""

from __future__ import annotations

import os
import sys


def configure_stdio() -> None:
    """Keep Cyrillic logs from crashing on Windows cp1252/charmap consoles."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_stdio()


def gui_mode() -> bool:
    if getattr(sys, "frozen", False):
        return True
    stdin = sys.stdin
    if stdin is None:
        return True
    try:
        return not stdin.isatty()
    except Exception:
        return True


def _windows_message(title: str, message: str, flags: int) -> int:
    import ctypes

    return int(ctypes.windll.user32.MessageBoxW(None, str(message), str(title), flags))


def show_error(title: str, message: str) -> None:
    if sys.platform == "win32":
        _windows_message(title, message, 0x00000010 | 0x00040000)  # ICONERROR | TOPMOST
        return
    if gui_mode():
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror(title, message, parent=root)
            root.destroy()
            return
        except Exception:
            pass
    print(f"{title}: {message}", file=sys.stderr)


def show_info(title: str, message: str) -> None:
    if sys.platform == "win32":
        _windows_message(title, message, 0x00000040 | 0x00040000)  # ICONINFORMATION | TOPMOST
        return
    if gui_mode():
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message, parent=root)
            root.destroy()
            return
        except Exception:
            pass
    print(f"{title}: {message}")


def ask_yes_no(title: str, message: str, default: bool = False) -> bool:
    if sys.platform == "win32":
        flags = 0x00000004 | 0x00000020 | 0x00040000  # YESNO | ICONQUESTION | TOPMOST
        if default:
            flags |= 0x00000000  # MB_DEFBUTTON1 = Yes
        else:
            flags |= 0x00000100  # MB_DEFBUTTON2 = No
        return _windows_message(title, message, flags) == 6  # IDYES
    if gui_mode():
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            answer = messagebox.askyesno(title, message, parent=root, default="yes" if default else "no")
            root.destroy()
            return bool(answer)
        except Exception:
            pass
    if not sys.stdin.isatty():
        return default
    print(message)
    while True:
        answer = input("Согласны? [y/N]: ").strip().lower()
        if answer in {"y", "yes", "д", "да"}:
            return True
        if answer in {"n", "no", "н", "нет", ""}:
            return False


def exit_with_error(message: str, code: int = 1) -> None:
    show_error("IUP", message)
    raise SystemExit(code)


def report_crash(error: BaseException) -> None:
    import traceback

    details = "".join(traceback.format_exception(error))
    log_path = None
    try:
        from agent.paths import writable_root

        log_path = writable_root() / "agent.log"
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(details)
            if not details.endswith("\n"):
                handle.write("\n")
    except Exception:
        pass
    extra = f"\n\nЛог: {log_path}" if log_path else ""
    exit_with_error(f"{error}{extra}")
