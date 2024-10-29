import cv2
import mediapipe as mp
import pyaudio
import audioop
import time
import telebot
import numpy as np

# Инициализация телеграм-бота
bot_token = '7812033794:AAHyQjmnQdyji8kQpzy8O-PSdY0IqtLLFV0'
chat_id = '-4584813855'
bot = telebot.TeleBot(bot_token)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
p = pyaudio.PyAudio()

# Telegram уведомление
def send_telegram_message(message):
    try:
        bot.send_message(chat_id, message)
    except Exception as e:
        print(f"Ошибка при отправке сообщения в Telegram: {e}")

# Проверка громкости
def check_microphone_volume(stream):
    data = stream.read(1024)
    rms = audioop.rms(data, 2)
    if rms < 50:
        send_telegram_message("Внимание: слишком тихий звук.")
    elif rms > 300:
        send_telegram_message("Внимание: слишком громкий звук.")
    return rms

def main():
    cap = cv2.VideoCapture(0)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    last_check_time = 0
    check_interval = 3

    last_angle = None
    last_faces_message_time = 0
    last_angle_message_time = 0
    message_interval = 10

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Не удалось получить кадр с камеры.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            current_time = time.time()
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                num_faces = len(results.multi_face_landmarks)
                if num_faces > 1 and (current_time - last_faces_message_time >= message_interval):
                    send_telegram_message(f"Обнаружено {num_faces} лица в кадре!")
                    last_faces_message_time = current_time

                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

                    left_eye = np.array([face_landmarks.landmark[362].x, face_landmarks.landmark[362].y])
                    right_eye = np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
                    nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y])

                    eye_line_vector = right_eye - left_eye
                    angle = np.arctan2(eye_line_vector[1], eye_line_vector[0]) * (180 / np.pi)

                    if last_angle is None or abs(angle - last_angle) > 10:
                        if current_time - last_angle_message_time >= message_interval:
                            send_telegram_message(f'Поворот головы: {angle:.2f}°')
                            last_angle_message_time = current_time
                        last_angle = angle

            cv2.imshow('Face Tracking', image)

            if current_time - last_check_time >= check_interval:
                check_microphone_volume(stream)
                last_check_time = current_time

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    stream.stop_stream()
    stream.close()

if __name__ == "__main__":
    main()
