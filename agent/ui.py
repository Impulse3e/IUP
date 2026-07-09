"""Shared GUI helpers for frozen / windowed builds."""

from __future__ import annotations

import sys


def gui_mode() -> bool:
    return getattr(sys, "frozen", False) or not sys.stdin.isatty()


def show_error(title: str, message: str) -> None:
    if gui_mode():
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror(title, message, parent=root)
            root.destroy()
        except Exception:
            print(f"{title}: {message}", file=sys.stderr)
    else:
        print(f"{title}: {message}", file=sys.stderr)


def show_info(title: str, message: str) -> None:
    if gui_mode():
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message, parent=root)
            root.destroy()
        except Exception:
            print(f"{title}: {message}")
    else:
        print(f"{title}: {message}")


def ask_yes_no(title: str, message: str, default: bool = False) -> bool:
    if gui_mode():
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            answer = messagebox.askyesno(title, message, parent=root, default="yes" if default else "no")
            root.destroy()
            return answer
        except Exception:
            pass
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
