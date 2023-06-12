import cv2

def face_recognise(image):

    # Загрузка обученной модели распознавания лиц из библиотеки OpenCV
    face_recognizer = cv2.face.FisherFaceRecognizer_create()
    face_recognizer.read('trained_model.xml')

    # Загрузка изображения, на котором нужно распознать лица
    img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

    # Задание детектора лиц и его параметров
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade_params = {
        'scaleFactor': 1.2,
        'minNeighbors': 5,
        'minSize': (30, 30),
    }

    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(img, **face_cascade_params)

    # Список для хранения координат прямоугольников, охватывающих каждое обнаруженное лицо
    face_rects = []

    # Проход по каждому обнаруженному лицу
    for (x, y, w, h) in faces:
        # Вырезаем область с лицом из исходного изображения
        face_roi = img[y:y + h, x:x + w]

        # Нормализация изображения перед подачей на вход модели
        face_roi_normalized = cv2.equalizeHist(face_roi)

        # Применение модели для распознавания лица
        label, confidence = face_recognizer.predict(face_roi_normalized)

        # Добавление координат прямоугольника, охватывающего текущее лицо, в список
        face_rects.append((x, y, w, h))

    # Возвращение списка координат прямоугольников, охватывающих каждое обнаруженное лицо
    return face_rects

