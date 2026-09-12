from datetime import datetime, timezone
from typing import Any


def parse_exam_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def exam_phase(exam) -> str:
    settings = getattr(exam, "settings", None) or {}
    now = datetime.utcnow()
    opens = parse_exam_time(settings.get("opens_at"))
    closes = parse_exam_time(settings.get("closes_at"))
    if opens and now < opens:
        return "before"
    if closes and now > closes:
        return "after"
    return "open"


def exam_window_message(exam) -> str | None:
    phase = exam_phase(exam)
    if phase == "before":
        return "Экзамен ещё не начался"
    if phase == "after":
        return "Окно экзамена закрыто"
    return None
