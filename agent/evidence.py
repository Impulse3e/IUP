from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class EvidenceBuffer:
    def __init__(self, seconds: int = 10, fps: int = 20) -> None:
        self.maxlen = seconds * fps
        self.frames: deque[np.ndarray] = deque(maxlen=self.maxlen)

    def push(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def save_clip(self, directory: Path, prefix: str) -> str | None:
        if not self.frames:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        height, width = self.frames[0].shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = directory / f"{prefix}_{timestamp}.avi"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"XVID"), 20.0, (width, height))
        for frame in self.frames:
            writer.write(frame)
        writer.release()
        return str(output)
