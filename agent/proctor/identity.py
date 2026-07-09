import numpy as np


def estimate_head_pose(landmarks):
    from agent.proctor.engine import HeadPose

    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    nose = np.array([landmarks[1].x, landmarks[1].y])
    chin = np.array([landmarks[152].x, landmarks[152].y])
    forehead = np.array([landmarks[10].x, landmarks[10].y])

    eye_mid = (left_eye + right_eye) / 2
    face_width = max(float(np.linalg.norm(right_eye - left_eye)), 1e-6)
    face_height = max(float(np.linalg.norm(chin - forehead)), 1e-6)

    yaw = ((nose[0] - eye_mid[0]) / face_width) * 90
    pitch = ((nose[1] - eye_mid[1]) / face_height - 0.35) * 90
    eye_vector = right_eye - left_eye
    roll = float(np.arctan2(eye_vector[1], eye_vector[0]) * (180 / np.pi))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


def face_embedding(landmarks) -> list[float]:
    indices = (33, 133, 362, 263, 1, 61, 291, 152, 10, 234, 454)
    vector = []
    for index in indices:
        vector.extend([landmarks[index].x, landmarks[index].y, landmarks[index].z])
    arr = np.array(vector, dtype=np.float64)
    arr -= arr.mean()
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr.tolist()


def compare_embeddings(left: list[float], right: list[float]) -> float:
    a = np.array(left, dtype=np.float64)
    b = np.array(right, dtype=np.float64)
    return float(np.linalg.norm(a - b))
