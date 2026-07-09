from enum import Enum


class ViolationType(str, Enum):
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    LOOK_AWAY = "look_away"
    AUDIO_QUIET = "audio_quiet"
    AUDIO_LOUD = "audio_loud"
    IDENTITY_MISMATCH = "identity_mismatch"
    FORBIDDEN_PROCESS = "forbidden_process"
    WINDOW_FOCUS_LOST = "window_focus_lost"
    AGENT_TAMPER = "agent_tamper"
    HEARTBEAT_LOST = "heartbeat_lost"
    SECOND_MONITOR = "second_monitor"
    PRE_CHECK_FAILED = "pre_check_failed"


VIOLATION_LABELS = {
    ViolationType.NO_FACE: "лицо вне кадра",
    ViolationType.MULTIPLE_FACES: "несколько лиц в кадре",
    ViolationType.LOOK_AWAY: "отведён взгляд / поворот головы",
    ViolationType.AUDIO_QUIET: "слишком тихий звук",
    ViolationType.AUDIO_LOUD: "слишком громкий звук",
    ViolationType.IDENTITY_MISMATCH: "несовпадение лица",
    ViolationType.FORBIDDEN_PROCESS: "запрещённый процесс",
    ViolationType.WINDOW_FOCUS_LOST: "потеря фокуса окна",
    ViolationType.AGENT_TAMPER: "вмешательство в агент",
    ViolationType.HEARTBEAT_LOST: "агент не отвечает",
    ViolationType.SECOND_MONITOR: "второй монитор",
    ViolationType.PRE_CHECK_FAILED: "не пройден pre-check",
}


class UserRole(str, Enum):
    STUDENT = "student"
    PROCTOR = "proctor"
    TEACHER = "teacher"
    ADMIN = "admin"


class SessionStatus(str, Enum):
    PENDING = "pending"
    PRECHECK = "precheck"
    ACTIVE = "active"
    COMPLETED = "completed"
    COMPROMISED = "compromised"
    CANCELLED = "cancelled"


class ReviewDecision(str, Enum):
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    INVALIDATE = "invalidate"


FORBIDDEN_PROCESSES = {
    "discord", "telegram", "zoom", "teams", "skype", "obs", "obs64",
    "anydesk", "teamviewer", "rustdesk", "chrome", "firefox", "msedge",
    "vivaldi", "opera", "virtualcam", "manycam", "snapcamera",
}
