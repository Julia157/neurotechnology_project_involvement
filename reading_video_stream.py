import sys
import cv2
import pygetwindow as gw
import numpy as np
from mss import mss
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json  # Используйте model_from_json для загрузки архитектуры модели

# Загрузка предварительно обученного детектора лиц Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка архитектуры модели эмоций из JSON-файла
with open('C:/Users/julia/Downloads/нейро_техчасть-20231126T114046Z-001/нейро_техчасть/project/model.json', 'r') as json_file:
    model_json = json_file.read()
    emotion_model = model_from_json(model_json)

# Загрузка весов модели эмоций
emotion_model.load_weights('C:/Users/julia/Downloads/нейро_техчасть-20231126T114046Z-001/нейро_техчасть/project/model.h5')

# Эмоциональные ярлыки, соответствующие выводу модели
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Получение окна Zoom Meeting
zoom_window_title = 'Zoom Meeting'
zoom_window = gw.getWindowsWithTitle(zoom_window_title)[0]

if not zoom_window:
    print("Zoom window not found")
    sys.exit()

zoom_window.activate()

with mss() as sct:
    while True:
        try:
            # Получение позиции и размеров окна Zoom
            left, top, width, height = zoom_window.left, zoom_window.top, zoom_window.width, zoom_window.height

            # Захват изображения для региона окна Zoom
            screen_image = sct.grab({"left": left, "top": top, "width": width, "height": height})
            frame = np.array(screen_image)

            # Создание копии оригинального кадра
            frame_copy = frame.copy()

            # Преобразование цветового пространства из RGB в BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Преобразование изображения в оттенки серого для обнаружения лиц
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Обнаружение лиц на изображении
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Инициализация словаря для хранения количества эмоций
            emotion_counts = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

            # Обработка каждого обнаруженного лица
            for (x, y, w, h) in faces:
                # Извлечение области лица
                face_roi = gray[y:y+h, x:x+w]

                # Изменение размера изображения лица для соответствия входному размеру модели эмоций
                face_roi_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

                # Нормализация значений пикселей в диапазоне [0, 1]
                face_roi_normalized = face_roi_resized / 255.0

                # Изменение формы изображения для соответствия ожидаемой форме входа модели
                face_roi_reshaped = np.reshape(face_roi_normalized, (1, 48, 48, 1))

                # Предсказание эмоции с использованием модели
                emotion_probabilities = emotion_model.predict(face_roi_reshaped)
                predicted_emotion_index = np.argmax(emotion_probabilities)

                # Получение предсказанной эмоциональной метки
                predicted_emotion = emotion_labels[predicted_emotion_index]

                # Увеличение счетчика обнаруженной эмоции
                emotion_counts[predicted_emotion] += 1

                # Рисование прямоугольника вокруг лица
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Отображение предсказанной эмоциональной метки
                cv2.putText(frame_copy, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Рисование счетчика эмоций в верхнем левом углу
            text_y_pos = 70  # Сброс позиции по y для каждого кадра
            for key in emotion_counts.keys():
                count_text = f"{key}: {emotion_counts[key]}"

                # Создание прозрачного оверлея для текста
                overlay = frame_copy.copy()
                cv2.putText(overlay, count_text, (60, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Смешивание оверлея с оригинальным кадром с использованием прозрачности
                cv2.addWeighted(overlay, 0.5, frame_copy, 0.5, 0, frame_copy)

                text_y_pos += 30
            
            # Определение эмоции с максимальным количеством
            max_emotion = max(emotion_counts, key=emotion_counts.get)

            # Расчет соотношения максимального количества эмоции к общему количеству распознанных эмоций
            total_recognized_emotions = sum(emotion_counts.values())
            ratio_max_emotion = emotion_counts[max_emotion] / total_recognized_emotions if total_recognized_emotions > 0 else 0
            cv2.putText(overlay, f"Involvement: {round(ratio_max_emotion, 2)}", (60, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.5, frame_copy, 0.5, 0, frame_copy)

            # Отображение результата с оверлеем
            cv2.imshow('Zoom Meeting Capture with Emotion Detection', frame_copy)

            # Прерывание цикла при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")

# Закрытие всех окон
cv2.destroyAllWindows()