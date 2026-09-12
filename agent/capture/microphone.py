import numpy as np


def calculate_rms(audio_chunk: bytes) -> float:
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    if audio_data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
