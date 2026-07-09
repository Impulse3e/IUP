import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyaudio

from agent.capture.microphone import calculate_rms
from agent.capture.screen import screenshot_bgr
from agent.capture.process_monitor import list_forbidden_processes, monitor_count
from agent.config import AgentConfig
from agent.evidence import EvidenceBuffer
from agent.precheck import ensure_precheck
from agent.proctor.engine import ProctorConfig, ProctoringEngine
from agent.proctor.face import create_face_landmarker, ensure_model
from agent.proctor.identity import compare_embeddings, face_embedding
from agent.proctor.overlay import draw_face_landmarks, draw_overlay
from agent.security.consent import ask_consent
from agent.security.tamper import TamperGuard
from agent.transport.client import SessionClient
from agent.ui import exit_with_error, gui_mode, show_info
from shared.constants import ViolationType
from shared.types import ProctorEvent

AGENT_VERSION = "2.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IUP Proctoring Agent")
    parser.add_argument("--token", help="Session access token")
    parser.add_argument("--server", help="Server URL")
    return parser.parse_args()


def capture_identity(cap, landmarker) -> tuple[list[float], int]:
    frame_timestamp_ms = 0
    for _ in range(30):
        success, frame = cap.read()
        if not success:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 50
        if result.face_landmarks:
            return face_embedding(result.face_landmarks[0]), frame_timestamp_ms
        cv2.imshow("Identity Check", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break
    raise RuntimeError("Не удалось получить эталонное лицо")


def main() -> None:
    args = parse_args()
    config = AgentConfig()
    if args.token:
        config.session_token = args.token
    if args.server:
        config.server_url = args.server
    if not config.session_token:
        exit_with_error("Укажите --token или IUP_SESSION_TOKEN в .env")

    client = SessionClient(config.server_url, config.session_token)
    session = client.fetch_session()
    if not gui_mode():
        print(f"Сессия: {session['id']} | экзамен: {session['exam_id']}")

    if not ask_consent():
        client.accept_consent(False)
        exit_with_error("Согласие не получено.")
    client.accept_consent(True)

    ensure_precheck()
    model_path = ensure_model(config.model_dir)
    landmarker = create_face_landmarker(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit_with_error("Не удалось открыть веб-камеру.")

    if not gui_mode():
        print("Идентификация лица...")
    try:
        reference, frame_timestamp_ms = capture_identity(cap, landmarker)
    except RuntimeError as error:
        exit_with_error(str(error))
    client.submit_identity(reference)
    client.start()

    proctor_config = ProctorConfig(
        yaw_threshold=config.yaw_threshold,
        pitch_threshold=config.pitch_threshold,
        roll_threshold=config.roll_threshold,
        confirm_frames=config.confirm_frames,
        alert_cooldown=config.alert_cooldown,
        reminder_interval=config.reminder_interval,
        audio_quiet_rms=config.audio_quiet_rms,
        audio_loud_rms=config.audio_loud_rms,
        audio_sustain_sec=config.audio_sustain_sec,
        audio_check_interval=config.audio_check_interval,
    )
    tamper = TamperGuard()
    evidence_dir = config.data_dir / "evidence"
    evidence_buffer = EvidenceBuffer(seconds=config.evidence_seconds)
    last_violation_upload = {}
    chunk_index = {"webcam": 0, "screen": 0}
    last_chunk_at = 0.0
    last_heartbeat = 0.0
    last_process_check = 0.0
    reference_embedding = reference

    def on_event(event: ProctorEvent) -> None:
        if tamper.check_debugger():
            tamper.mark_compromised("debugger")
        try:
            result = client.post_event(event)
            if not event.is_resolved and not event.is_reminder:
                clip = evidence_buffer.save_clip(evidence_dir, event.type)
                if clip and event.type not in last_violation_upload:
                    client.upload_evidence("video_clip", clip, violation_id=result.get("id"))
                    last_violation_upload[event.type] = time.time()
        except Exception as error:
            print(f"Ошибка отправки события: {error}")

    engine = ProctoringEngine(config=proctor_config, on_event=on_event)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    def heartbeat_loop() -> None:
        while True:
            if tamper.compromised:
                client.heartbeat(AGENT_VERSION, status="compromised", payload=tamper.status())
            else:
                client.heartbeat(AGENT_VERSION, payload=tamper.status())
            time.sleep(config.heartbeat_interval)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            evidence_buffer.push(image)
            frame_screen = screenshot_bgr()

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            now = time.time()
            frame_timestamp_ms += 50
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            face_landmarks_list = results.face_landmarks or []

            engine.analyze_faces(face_landmarks_list, now)
            if face_landmarks_list:
                current_embedding = face_embedding(face_landmarks_list[0])
                distance = compare_embeddings(reference_embedding, current_embedding)
                if distance > config.identity_threshold:
                    engine.report_custom(
                        ViolationType.IDENTITY_MISMATCH,
                        f"лицо не совпадает с эталоном (distance {distance:.3f}).",
                        severity="critical",
                    )
                for face_landmarks in face_landmarks_list:
                    draw_face_landmarks(image, face_landmarks)

            audio_chunk = stream.read(1024, exception_on_overflow=False)
            engine.analyze_audio(calculate_rms(audio_chunk), now)

            if now - last_process_check >= 10:
                forbidden = list_forbidden_processes()
                if forbidden:
                    engine.report_custom(
                        ViolationType.FORBIDDEN_PROCESS,
                        f"обнаружены процессы: {', '.join(forbidden)}",
                        severity="high",
                    )
                if monitor_count() > 1:
                    engine.report_custom(
                        ViolationType.SECOND_MONITOR,
                        "обнаружен второй монитор.",
                        severity="high",
                    )
                last_process_check = now

            if now - last_chunk_at >= config.chunk_interval:
                _, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                client.upload_chunk("webcam", chunk_index["webcam"], encoded.tobytes())
                chunk_index["webcam"] += 1
                _, encoded_screen = cv2.imencode(".jpg", frame_screen, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                client.upload_chunk("screen", chunk_index["screen"], encoded_screen.tobytes())
                chunk_index["screen"] += 1
                last_chunk_at = now

            extra = ""
            if tamper.compromised:
                extra = "AGENT COMPROMISED"
            draw_overlay(image, engine, extra=extra)
            cv2.imshow("IUP Proctoring", image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
    finally:
        summary = engine.session_summary()
        client.end(summary)
        cap.release()
        stream.stop_stream()
        stream.close()
        audio.terminate()
        landmarker.close()
        client.close()
        cv2.destroyAllWindows()
        if not gui_mode():
            print("Сессия завершена.")
        else:
            show_info("IUP", "Сессия завершена.")


if __name__ == "__main__":
    main()
