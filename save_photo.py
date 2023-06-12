import cv2 as cv
folder = "video_2023-04-05_22-42-06"
capture = cv.VideoCapture(f'input/{folder}.mov')

i = 0
while True:
    ret, frame = capture.read()
    # find_contours_etalon(frame.copy())
    if not ret:
        break

    cv.imwrite(f'{folder}/img_{i}.jpg', frame)
    i += 1
