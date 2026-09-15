from pathlib import Path

import numpy as np


def frames_to_jpeg(frames: list, quality: int = 80, thumb_width: int = 320) -> bytes | None:
    if not frames:
        return None
    import cv2

    thumbs: list[np.ndarray] = []
    for frame in frames[:6]:
        if frame is None or getattr(frame, "size", 0) == 0:
            continue
        height, width = frame.shape[:2]
        scale = thumb_width / max(width, 1)
        thumbs.append(cv2.resize(frame, (thumb_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return None

    def pad_height(image: np.ndarray, target: int) -> np.ndarray:
        if image.shape[0] >= target:
            return image
        return cv2.copyMakeBorder(image, 0, target - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    def pad_width(image: np.ndarray, target: int) -> np.ndarray:
        if image.shape[1] >= target:
            return image
        return cv2.copyMakeBorder(image, 0, 0, 0, target - image.shape[1], cv2.BORDER_CONSTANT)

    height = max(item.shape[0] for item in thumbs)
    thumbs = [pad_height(item, height) for item in thumbs]
    if len(thumbs) <= 3:
        mosaic = np.hstack(thumbs)
    else:
        mid = (len(thumbs) + 1) // 2
        row1 = np.hstack(thumbs[:mid])
        row2 = np.hstack(thumbs[mid:])
        width = max(row1.shape[1], row2.shape[1])
        mosaic = np.vstack([pad_width(row1, width), pad_width(row2, width)])
    ok, buffer = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def video_file_to_jpeg(path: Path) -> bytes | None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        if total <= 0:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        else:
            indexes = sorted({int(total * index / 5) for index in range(6) if total * index / 5 < total})
            for index in indexes:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        return frames_to_jpeg(frames)
    finally:
        cap.release()
