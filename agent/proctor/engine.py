import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from shared.constants import VIOLATION_LABELS, ViolationType
from shared.types import ProctorEvent

from agent.proctor.identity import estimate_head_pose


@dataclass
class ProctorConfig:
    yaw_threshold: float = 25.0
    pitch_threshold: float = 45.0
    roll_threshold: float = 15.0
    confirm_frames: int = 15
    alert_cooldown: float = 30.0
    reminder_interval: float = 60.0
    audio_quiet_rms: float = 50.0
    audio_loud_rms: float = 300.0
    audio_sustain_sec: float = 2.0
    audio_check_interval: float = 0.5


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float

    def is_away(self, config: ProctorConfig) -> bool:
        return (
            abs(self.yaw) > config.yaw_threshold
            or abs(self.pitch) > config.pitch_threshold
            or abs(self.roll) > config.roll_threshold
        )


@dataclass
class ViolationState:
    active: bool = False
    streak: int = 0
    last_alert_time: float = 0.0
    count: int = 0


@dataclass
class ProctoringEngine:
    config: ProctorConfig
    on_event: Callable[[ProctorEvent], None]
    session_start: float = field(default_factory=time.time)
    violations: dict[ViolationType, ViolationState] = field(
        default_factory=lambda: {v: ViolationState() for v in ViolationType}
    )
    violation_totals: dict[ViolationType, int] = field(default_factory=lambda: defaultdict(int))
    current_pose: HeadPose | None = None
    face_count: int = 0
    audio_rms: float = 0.0
    audio_condition_since: float | None = None
    audio_condition: str | None = None
    last_audio_check: float = 0.0

    def _emit(
        self,
        violation: ViolationType,
        message: str,
        *,
        severity: str = "warning",
        payload: dict | None = None,
        is_reminder: bool = False,
        is_resolved: bool = False,
    ) -> None:
        self.on_event(
            ProctorEvent(
                type=violation.value,
                message=message,
                severity=severity,
                payload=payload or {},
                is_reminder=is_reminder,
                is_resolved=is_resolved,
            )
        )

    def _can_alert(self, violation: ViolationType, now: float, is_reminder: bool) -> bool:
        state = self.violations[violation]
        interval = self.config.reminder_interval if is_reminder else self.config.alert_cooldown
        return now - state.last_alert_time >= interval

    def _raise(self, violation: ViolationType, message: str, now: float, is_reminder: bool = False) -> None:
        state = self.violations[violation]
        if not self._can_alert(violation, now, is_reminder):
            return
        prefix = "Напоминание" if is_reminder else "Нарушение"
        self._emit(violation, f"{prefix}: {message}", is_reminder=is_reminder)
        state.last_alert_time = now
        state.count += 1
        self.violation_totals[violation] += 1

    def _resolve(self, violation: ViolationType, now: float) -> None:
        state = self.violations[violation]
        if not state.active:
            return
        state.active = False
        state.streak = 0
        self._emit(
            violation,
            f"Норма восстановлена: {VIOLATION_LABELS[violation]}.",
            is_resolved=True,
        )
        state.last_alert_time = now

    def _track_streak(
        self,
        violation: ViolationType,
        condition: bool,
        now: float,
        alert_message: str,
        reminder_message: str | None = None,
        severity: str = "warning",
    ) -> None:
        state = self.violations[violation]
        if condition:
            state.streak += 1
            if state.streak >= self.config.confirm_frames and not state.active:
                state.active = True
                self._raise(violation, alert_message, now)
            elif state.active and reminder_message and self._can_alert(violation, now, is_reminder=True):
                self._raise(violation, reminder_message, now, is_reminder=True)
        else:
            if state.active:
                self._resolve(violation, now)
            state.streak = 0

    def analyze_faces(self, face_landmarks_list, now: float) -> None:
        self.face_count = len(face_landmarks_list)
        self._track_streak(
            ViolationType.NO_FACE,
            self.face_count == 0,
            now,
            "лицо не обнаружено в кадре.",
            "лицо по-прежнему не видно.",
            severity="high",
        )
        self._track_streak(
            ViolationType.MULTIPLE_FACES,
            self.face_count > 1,
            now,
            f"обнаружено {self.face_count} лица в кадре.",
            f"в кадре по-прежнему {self.face_count} лица.",
            severity="high",
        )
        if self.face_count == 1:
            self.current_pose = estimate_head_pose(face_landmarks_list[0])
            pose = self.current_pose
            detail = f"yaw {pose.yaw:+.0f}°, pitch {pose.pitch:+.0f}°, roll {pose.roll:+.0f}°."
            self._track_streak(
                ViolationType.LOOK_AWAY,
                pose.is_away(self.config),
                now,
                f"голова отведена от экрана ({detail})",
                f"взгляд по-прежнему отведён ({detail})",
            )
        else:
            self.current_pose = None
            if self.violations[ViolationType.LOOK_AWAY].active:
                self._resolve(ViolationType.LOOK_AWAY, now)
            self.violations[ViolationType.LOOK_AWAY].streak = 0

    def analyze_audio(self, rms: float, now: float) -> None:
        if now - self.last_audio_check < self.config.audio_check_interval:
            return
        self.last_audio_check = now
        self.audio_rms = rms

        if rms < self.config.audio_quiet_rms:
            condition, violation, other = "quiet", ViolationType.AUDIO_QUIET, ViolationType.AUDIO_LOUD
        elif rms > self.config.audio_loud_rms:
            condition, violation, other = "loud", ViolationType.AUDIO_LOUD, ViolationType.AUDIO_QUIET
        else:
            if self.violations[ViolationType.AUDIO_QUIET].active:
                self._resolve(ViolationType.AUDIO_QUIET, now)
            if self.violations[ViolationType.AUDIO_LOUD].active:
                self._resolve(ViolationType.AUDIO_LOUD, now)
            self.audio_condition = None
            self.audio_condition_since = None
            return

        if self.audio_condition != condition:
            self.audio_condition = condition
            self.audio_condition_since = now
            if self.violations[other].active:
                self._resolve(other, now)
            return

        if self.audio_condition_since is None or now - self.audio_condition_since < self.config.audio_sustain_sec:
            return

        state = self.violations[violation]
        label = "тихий" if condition == "quiet" else "громкий"
        if not state.active:
            state.active = True
            self._raise(violation, f"микрофон слишком {label} (RMS {rms:.0f}).", now)
        else:
            self._raise(violation, f"звук по-прежнему слишком {label} (RMS {rms:.0f}).", now, is_reminder=True)

    def report_custom(self, violation: ViolationType, message: str, severity: str = "high") -> None:
        now = time.time()
        state = self.violations[violation]
        if not state.active:
            state.active = True
            self._raise(violation, message, now)
        elif self._can_alert(violation, now, is_reminder=True):
            self._raise(violation, message, now, is_reminder=True)
        self.violation_totals[violation] += 1

    def status_label(self) -> tuple[str, tuple[int, int, int]]:
        if any(self.violations[v].active for v in ViolationType):
            return "НАРУШЕНИЕ", (0, 0, 255)
        if any(0 < self.violations[v].streak < self.config.confirm_frames for v in ViolationType):
            return "ПРОВЕРКА", (0, 165, 255)
        return "НОРМА", (0, 200, 0)

    def session_summary(self) -> dict:
        duration = int(time.time() - self.session_start)
        totals = {v.value: c for v, c in self.violation_totals.items() if c}
        risk = min(100.0, sum(totals.values()) * 5)
        return {"duration_sec": duration, "violation_totals": totals, "risk_score": risk}
