import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from agent.paths import model_dir, project_root, writable_root

BASE_DIR = project_root()
load_dotenv(BASE_DIR / ".env")


@dataclass
class AgentConfig:
    server_url: str = os.getenv("IUP_SERVER_URL", "http://localhost:8000")
    session_token: str = os.getenv("IUP_SESSION_TOKEN", "")
    agent_version: str = "2.0.0"
    heartbeat_interval: float = float(os.getenv("IUP_HEARTBEAT_INTERVAL", 5))
    chunk_interval: float = float(os.getenv("IUP_CHUNK_INTERVAL", 30))
    evidence_seconds: int = int(os.getenv("IUP_EVIDENCE_SECONDS", 10))
    identity_threshold: float = float(os.getenv("IUP_IDENTITY_THRESHOLD", 0.15))
    yaw_threshold: float = float(os.getenv("PROCTOR_YAW_THRESHOLD", 25))
    pitch_threshold: float = float(os.getenv("PROCTOR_PITCH_THRESHOLD", 45))
    roll_threshold: float = float(os.getenv("PROCTOR_ROLL_THRESHOLD", 15))
    confirm_frames: int = int(os.getenv("PROCTOR_CONFIRM_FRAMES", 15))
    alert_cooldown: float = float(os.getenv("PROCTOR_ALERT_COOLDOWN", 30))
    reminder_interval: float = float(os.getenv("PROCTOR_REMINDER_INTERVAL", 60))
    audio_quiet_rms: float = float(os.getenv("PROCTOR_AUDIO_QUIET_RMS", 50))
    audio_loud_rms: float = float(os.getenv("PROCTOR_AUDIO_LOUD_RMS", 300))
    audio_sustain_sec: float = float(os.getenv("PROCTOR_AUDIO_SUSTAIN_SEC", 2))
    audio_check_interval: float = float(os.getenv("PROCTOR_AUDIO_CHECK_INTERVAL", 0.5))
    model_dir: Path = field(default_factory=model_dir)
    data_dir: Path = field(default_factory=lambda: writable_root() / "data" / "agent")
