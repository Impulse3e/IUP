import cv2
import mediapipe as mp
from gevent import monkey; monkey.patch_all()
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

# Инициализация Flask и SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Инициализация камеры
cap = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

def detect_face_and_send_updates():
    """Функция для отслеживания лица и отправки данных через сокеты."""
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Не удалось получить изображение с камеры.")
                break

            # Преобразуем изображение в RGB для обработки
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Обработка изображения для отслеживания лица
            results = face_mesh.process(image)

            # Преобразуем изображение обратно в BGR для отображения
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Если найдены лица
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Рисуем ключевые точки на лице
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            # Отображаем изображение
            cv2.imshow('Face Mesh', image)

            # Отправляем данные через сокеты
            socketio.emit('face_update', {'status': 'face_detected'})

            # Нажатие клавиши 'q' завершает цикл
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


@socketio.on('connect')
def handle_connect():
    """Когда клиент подключается, запускаем трекинг лица."""
    print('Клиент подключен')
    thread = threading.Thread(target=detect_face_and_send_updates)
    thread.start()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, use_reloader=False, debug=True)
