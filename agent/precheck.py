import sys

import cv2
import pyaudio

from agent.capture.process_monitor import list_forbidden_processes, monitor_count
from agent.capture.screen import screen_available
from agent.ui import exit_with_error, gui_mode, show_info


def run_precheck() -> dict:
    results = {"camera": False, "microphone": False, "screen": False, "monitors": 1, "forbidden": []}
    cap = cv2.VideoCapture(0)
    results["camera"] = cap.isOpened()
    if results["camera"]:
        cap.release()

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        stream.close()
        results["microphone"] = True
    except Exception:
        results["microphone"] = False
    finally:
        audio.terminate()

    results["screen"] = screen_available()
    results["monitors"] = monitor_count()
    results["forbidden"] = list_forbidden_processes()
    return results


def ensure_precheck() -> None:
    if not gui_mode():
        print("Запуск pre-check...")
    results = run_precheck()
    if not gui_mode():
        print(f"  Камера: {'OK' if results['camera'] else 'FAIL'}")
        print(f"  Микрофон: {'OK' if results['microphone'] else 'FAIL'}")
        print(f"  Экран: {'OK' if results['screen'] else 'FAIL'}")
        print(f"  Мониторов: {results['monitors']}")
        if results["forbidden"]:
            print(f"  Запрещённые процессы: {', '.join(results['forbidden'])}")

    missing = []
    if not results["camera"]:
        missing.append("веб-камера")
    if not results["microphone"]:
        missing.append("микрофон")
    if not results["screen"]:
        missing.append("доступ к экрану")

    if missing:
        exit_with_error("Pre-check не пройден. Проверьте: " + ", ".join(missing))

    if results["monitors"] > 1:
        show_info("IUP", "Обнаружен второй монитор. Рекомендуется отключить дополнительные дисплеи.")
    if results["forbidden"]:
        exit_with_error("Закройте запрещённые приложения: " + ", ".join(results["forbidden"]))
