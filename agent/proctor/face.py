import time
import urllib.request
from pathlib import Path

from agent.paths import writable_root

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(model_dir: Path) -> Path:
    model_path = model_dir / "face_landmarker.task"
    if model_path.exists():
        return model_path

    cache_dir = writable_root() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "face_landmarker.task"
    if model_path.exists():
        return model_path

    print("Скачиваю модель MediaPipe Face Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def create_face_landmarker(model_path: Path):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)
