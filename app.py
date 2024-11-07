import cv2
import mediapipe as mp
import pyaudio
import audioop
import time
import telebot
import numpy as np
import pyautogui

# Инициализация телеграм-бота
bot_token = '7812033794:AAHyQjmnQdyji8kQpzy8O-PSdY0IqtLLFV0'
chat_id = '-4584813855'
bot = telebot.TeleBot(bot_token)

# Пути к папкам для сохранения видео
webcam_video_path = r'D:\pycharm\zxcproject\webcam_video.avi'
screen_video_path = r'D:\pycharm\zxcproject\screen_video.avi'

# Инициализация MediaPipe и PyAudio
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
    # Инициализация камеры и записи видео
    cap = cv2.VideoCapture(0)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    # Настройка для записи видео с камеры и экрана
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_webcam = cv2.VideoWriter(webcam_video_path, fourcc, 20.0, (640, 480))
    screen_size = pyautogui.size()
    out_screen = cv2.VideoWriter(screen_video_path, fourcc, 20.0, screen_size)

    last_check_time = 0
    check_interval = 3
    last_face_detected = True  # Флаг для отслеживания последнего состояния лица в кадре
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

            # Запись видео с камеры
            out_webcam.write(image)

            # Захват экрана
            screen_image = pyautogui.screenshot()
            frame_screen = np.array(screen_image)
            frame_screen = cv2.cvtColor(frame_screen, cv2.COLOR_BGR2RGB)
            out_screen.write(frame_screen)

            # Преобразование изображения для обработки
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            current_time = time.time()
            results = face_mesh.process(image_rgb)

            # Проверка на наличие лиц
            if not results.multi_face_landmarks:
                if last_face_detected:
                    send_telegram_message("Внимание: не обнаружено ни одного лица в кадре!")
                    last_face_detected = False
            else:
                last_face_detected = True  # Лицо обнаружено, сброс флага
                
                # Проверка количества лиц и поворота головы
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

            # Отображение изображения с трекингом лица
            cv2.imshow('Face Tracking', image)

            # Проверка громкости звука
            if current_time - last_check_time >= check_interval:
                check_microphone_volume(stream)
                last_check_time = current_time

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    out_webcam.release()
    out_screen.release()
    stream.stop_stream()
    stream.close()

if __name__ == "__main__":
    main()
