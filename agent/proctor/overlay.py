import cv2
import numpy as np

from agent.proctor.engine import ProctoringEngine


def draw_overlay(image: np.ndarray, engine: ProctoringEngine, extra: str = "") -> None:
    status, color = engine.status_label()
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 88), (30, 30, 30), -1)
    cv2.putText(image, f"Статус: {status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if engine.current_pose:
        pose = engine.current_pose
        pose_text = f"Yaw {pose.yaw:+.0f}  Pitch {pose.pitch:+.0f}  Roll {pose.roll:+.0f}"
        cv2.putText(image, pose_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(image, f"Лиц: {engine.face_count}", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    total = sum(engine.violation_totals.values())
    cv2.putText(image, f"Нарушений: {total}", (width - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if extra:
        cv2.putText(image, extra, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)


def draw_face_landmarks(image: np.ndarray, face_landmarks) -> None:
    height, width = image.shape[:2]
    for index in (33, 263, 1, 152, 10, 234, 454):
        landmark = face_landmarks[index]
        cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 3, (0, 255, 0), -1)
