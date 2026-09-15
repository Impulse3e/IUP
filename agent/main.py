import argparse
import queue
import sys
import threading
import time
from collections.abc import Callable

import cv2
import mediapipe as mp
import pyaudio

from agent.capture.camera import call_with_camera_pump, open_camera, show_camera_window
from agent.capture.focus import foreground_window, match_watch_title, window_focus_lost
from agent.capture.microphone import calculate_rms
from agent.capture.screen import screenshot_bgr
from agent.capture.process_monitor import list_forbidden_processes, monitor_count
from agent.config import AgentConfig
from agent.identity_store import load_embedding, save_embedding
from agent.evidence import EvidenceBuffer, encode_jpeg
from agent.precheck import ensure_precheck
from agent.proctor.engine import ProctorConfig, ProctoringEngine
from agent.proctor.face import create_face_landmarker, ensure_model
from agent.proctor.identity import compare_embeddings, face_embedding
from agent.proctor.overlay import draw_face_landmarks, draw_overlay
from agent.security.consent import ask_consent
from agent.security.tamper import TamperGuard
from agent.transport.client import SessionClient
from agent.ui import exit_with_error, gui_mode, report_crash, show_info
from shared.constants import ViolationType
from shared.types import ProctorEvent

AGENT_VERSION = "2.0.0"
SNAPSHOT_EVENT_TYPES = {
    ViolationType.FORBIDDEN_PROCESS.value,
    ViolationType.WINDOW_FOCUS_LOST.value,
    ViolationType.SECOND_MONITOR.value,
    ViolationType.WATCHED_TITLE.value,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IUP Proctoring Agent")
    parser.add_argument("--token", help="Session access token")
    parser.add_argument("--server", help="Server URL")
    parser.add_argument(
        "--consent-accepted",
        action="store_true",
        help="Согласие уже получено в лаунчере, не спрашивать повторно",
    )
    return parser.parse_args()


def capture_identity(cap, landmarker, timeout_sec: float = 20.0) -> tuple[list[float], int]:
    frame_timestamp_ms = 0
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        success, frame = cap.read()
        if not success:
            time.sleep(0.03)
            continue
        remaining = max(0, int(deadline - time.time()))
        prompt = f"Look at the camera  ({remaining}s)"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (30, 30, 30), -1)
        cv2.putText(frame, prompt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        show_camera_window(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33
        if result.face_landmarks:
            return face_embedding(result.face_landmarks[0]), frame_timestamp_ms
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    raise RuntimeError("Не удалось получить эталонное лицо. Сядьте напротив камеры и повторите запуск.")


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
    watch_titles = [
        str(item).strip().lower()
        for item in (session.get("watch_titles") or [])
        if str(item).strip()
    ]
    if session.get("status") in {"completed", "cancelled"}:
        exit_with_error("Сессия уже завершена.")
    if not gui_mode():
        print(f"Сессия: {session['id']} | экзамен: {session['exam_id']}")

    already_consented = bool(session.get("consent_at"))
    if not already_consented:
        accepted = bool(args.consent_accepted)
        if not accepted:
            accepted = ask_consent()
        if not accepted:
            client.accept_consent(False)
            exit_with_error("Согласие не получено.")
        client.accept_consent(True)

    ensure_precheck()
    model_path = ensure_model(config.model_dir)
    landmarker = create_face_landmarker(model_path)

    cap = open_camera()
    if not cap.isOpened():
        exit_with_error("Не удалось открыть веб-камеру. Закройте другие приложения, которые её используют.")

    warmup, _ = cap.read()
    if warmup is not None:
        show_camera_window(warmup)
        cv2.waitKey(1)

    session_id = session["id"]
    identity_verified = bool(session.get("identity_verified"))
    reference = load_embedding(session_id)
    frame_timestamp_ms = 0
    if identity_verified and reference:
        if not gui_mode():
            print("Идентификация уже пройдена, продолжаем сессию.")
    else:
        if not gui_mode():
            print("Идентификация лица: посмотрите в камеру...")
        try:
            reference, frame_timestamp_ms = capture_identity(cap, landmarker)
        except RuntimeError as error:
            exit_with_error(str(error))
        save_embedding(session_id, reference)

    def connect_exam() -> None:
        if not identity_verified:
            client.submit_identity(reference)
        save_embedding(session_id, reference)
        if session.get("status") != "active":
            client.start()

    try:
        call_with_camera_pump(cap, connect_exam, "Connecting to exam...")
    except RuntimeError as error:
        exit_with_error(str(error))

    proctor_config = ProctorConfig(
        yaw_threshold=config.yaw_threshold,
        pitch_threshold=config.pitch_threshold,
        roll_threshold=config.roll_threshold,
        confirm_frames=config.confirm_frames,
        focus_confirm_frames=config.focus_confirm_frames,
        identity_confirm_frames=config.identity_confirm_frames,
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
    last_preview_at = 0.0
    last_process_check = 0.0
    last_title_check = 0.0
    last_identity_at = 0.0
    reference_embedding = reference
    latest_frame: dict = {"image": None}
    jobs: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=24)
    frame_interval = 1 / 15

    def worker() -> None:
        while True:
            job = jobs.get()
            try:
                if job is None:
                    return
                job()
            except Exception as error:
                print(f"Ошибка фоновой задачи: {error}")
            finally:
                jobs.task_done()

    for _ in range(2):
        threading.Thread(target=worker, daemon=True).start()

    def put_job(job: Callable[[], None], droppable: bool = False) -> None:
        if droppable:
            try:
                jobs.put_nowait(job)
            except queue.Full:
                return
            return
        jobs.put(job)

    def on_event(event: ProctorEvent) -> None:
        if tamper.check_debugger():
            tamper.mark_compromised("debugger")
        should_snapshot = not event.is_resolved and event.type in SNAPSHOT_EVENT_TYPES
        webcam_bytes = encode_jpeg(latest_frame.get("image"), quality=60, max_width=640) if should_snapshot else None
        screen_bytes = None
        if should_snapshot:
            try:
                screen_bytes = encode_jpeg(
                    screenshot_bgr(prefer_foreground=event.type != ViolationType.SECOND_MONITOR.value),
                    quality=70,
                    max_width=1600,
                )
            except Exception as error:
                print(f"Ошибка снимка экрана: {error}")
        should_clip = (
            not event.is_resolved
            and not event.is_reminder
            and event.type not in SNAPSHOT_EVENT_TYPES
            and event.type not in last_violation_upload
        )
        frames = list(evidence_buffer.frames) if should_clip else []
        if should_clip:
            last_violation_upload[event.type] = time.time()

        def job(
            event=event,
            frames=frames,
            should_clip=should_clip,
            webcam_bytes=webcam_bytes,
            screen_bytes=screen_bytes,
        ) -> None:
            try:
                result = client.post_event(event)
                violation_id = result.get("id")
                if webcam_bytes:
                    client.upload_evidence_bytes("webcam_still", webcam_bytes, "webcam.jpg", violation_id)
                if screen_bytes:
                    client.upload_evidence_bytes("screen_still", screen_bytes, "screen.jpg", violation_id)
                if should_clip:
                    clip = evidence_buffer.save_clip(evidence_dir, event.type, frames=frames)
                    if clip:
                        client.upload_evidence("video_clip", clip, violation_id=violation_id)
            except Exception as error:
                print(f"Ошибка отправки события: {error}")

        put_job(job)

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
            loop_started = time.perf_counter()
            success, image = cap.read()
            if not success:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            latest_frame["image"] = image

            evidence_buffer.push(image)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            now = time.time()
            frame_timestamp_ms += 66
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            face_landmarks_list = results.face_landmarks or []

            engine.analyze_faces(face_landmarks_list, now)
            engine.analyze_focus(window_focus_lost(), now)
            if face_landmarks_list:
                if now - last_identity_at >= 0.4:
                    current_embedding = face_embedding(face_landmarks_list[0])
                    distance = compare_embeddings(reference_embedding, current_embedding)
                    engine.analyze_identity(
                        distance > config.identity_threshold,
                        now,
                        f"лицо не совпадает с эталоном (distance {distance:.3f}).",
                    )
                    last_identity_at = now
                draw_face_landmarks(image, face_landmarks_list[0])
            else:
                engine.analyze_identity(False, now, "")

            audio_chunk = stream.read(1024, exception_on_overflow=False)
            engine.analyze_audio(calculate_rms(audio_chunk), now)

            if watch_titles and now - last_title_check >= 2:
                active = foreground_window()
                matched = match_watch_title(active.get("title") or "", watch_titles)
                if matched:
                    title = (active.get("title") or "")[:120]
                    engine.report_custom(
                        ViolationType.WATCHED_TITLE,
                        f"в заголовке окна есть «{matched}»: {title}",
                        severity="warning",
                        payload={"title": title, "matched": matched, "process": active.get("process") or ""},
                    )
                else:
                    engine.clear_custom(ViolationType.WATCHED_TITLE)
                last_title_check = now

            if now - last_process_check >= 10:
                forbidden = list_forbidden_processes()
                if forbidden:
                    engine.report_custom(
                        ViolationType.FORBIDDEN_PROCESS,
                        f"обнаружены процессы: {', '.join(forbidden)}",
                        severity="high",
                        payload={"processes": forbidden},
                    )
                else:
                    engine.clear_custom(ViolationType.FORBIDDEN_PROCESS)
                if monitor_count() > 1:
                    engine.report_custom(
                        ViolationType.SECOND_MONITOR,
                        "обнаружен второй монитор.",
                        severity="high",
                    )
                else:
                    engine.clear_custom(ViolationType.SECOND_MONITOR)
                last_process_check = now

            if now - last_preview_at >= config.preview_interval:
                _, preview = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                preview_bytes = preview.tobytes()

                def upload_preview(preview_bytes=preview_bytes) -> None:
                    client.upload_chunk("preview", 0, preview_bytes)

                put_job(upload_preview, droppable=True)
                last_preview_at = now

            if now - last_chunk_at >= config.chunk_interval:
                _, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                webcam_bytes = encoded.tobytes()
                webcam_idx = chunk_index["webcam"]
                chunk_index["webcam"] += 1
                screen_idx = chunk_index["screen"]
                chunk_index["screen"] += 1

                def upload(
                    webcam_bytes=webcam_bytes,
                    webcam_idx=webcam_idx,
                    screen_idx=screen_idx,
                ) -> None:
                    client.upload_chunk("webcam", webcam_idx, webcam_bytes)
                    try:
                        frame_screen = screenshot_bgr()
                        _, encoded_screen = cv2.imencode(".jpg", frame_screen, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
                        client.upload_chunk("screen", screen_idx, encoded_screen.tobytes())
                    except Exception as error:
                        print(f"Ошибка снимка экрана: {error}")

                put_job(upload, droppable=True)
                last_chunk_at = now

            extra = ""
            if tamper.compromised:
                extra = "AGENT COMPROMISED"
            draw_overlay(image, engine, extra=extra)
            show_camera_window(image)

            leftover_ms = int((frame_interval - (time.perf_counter() - loop_started)) * 1000)
            if cv2.waitKey(max(1, leftover_ms)) & 0xFF == ord("q"):
                break
    finally:
        summary = engine.session_summary()
        jobs.join()
        try:
            client.end(summary)
        except Exception as error:
            print(f"Ошибка завершения сессии: {error}")
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
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        report_crash(error)
