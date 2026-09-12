"""HTTP helpers for the student desktop apps."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import httpx


def normalize_server_url(url: str) -> str:
    """Use IPv4 loopback so Windows does not stall on localhost → ::1."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return "http://127.0.0.1:8000"
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host != "localhost":
        return url
    netloc = parts.netloc
    idx = netloc.lower().find("localhost")
    if idx < 0:
        return url
    replaced = netloc[:idx] + "127.0.0.1" + netloc[idx + len("localhost") :]
    return urlunsplit(parts._replace(netloc=replaced)).rstrip("/")


def http_timeout(kind: str = "api") -> httpx.Timeout:
    if kind == "upload":
        return httpx.Timeout(30.0, connect=3.0)
    if kind == "heartbeat":
        return httpx.Timeout(5.0, connect=2.0)
    return httpx.Timeout(10.0, connect=3.0)
