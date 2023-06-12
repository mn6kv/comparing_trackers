import os
import numpy as np
import cv2 as cv
import time
import argparse
from itertools import islice
from src.tracker import *
from src.model_specific import *

if not os.path.exists('output'):
    os.makedirs('output')

MODEL_PATH = r'res/lbpcascade_frontalface.xml'

# Deep Learning
def face_recognise(img):
    prototxtPath = r'res/deploy.prototxt.txt'
    weightsPath = r'res/res10_300x300_ssd_iter_140000.caffemodel'
    face_cascade_db = cv.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    img_blob = cv.dnn.blobFromImage(cv.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_rects = []
    count = 0
    (h, w) = img.shape[:2]

    face_cascade_db.setInput(img_blob)
    detections = face_cascade_db.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.88:
            count += 1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.rectangle(img,
                    (startX, startY),
                    (endX, endY),
                    (0, 255, 255), 2)
            cv.putText(img, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            face_rects.append((startX, startY, endX - startX, endY - startY))
    print('Count ', count)

    if not face_rects:
        raise ValueError('Не обнаружено ни одного искомого объекта')

    return face_rects

# LBP
# def face_recognise(img):
#     face_cascade_db = cv.CascadeClassifier(MODEL_PATH)
#
#     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     face_rects = []
#
#     faces = face_cascade_db.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
#
#     for (x, y, w, h) in faces:
#         cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         face_rects.append((x, y, w, h))
#
#     if not face_rects:
#         raise ValueError('Не обнаружено ни одного искомого объекта')
#
#     return face_rects

# Каскады Хаара
# def face_recognise(img):
#     face_cascade_db = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
#
#     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     face_rects = []
#
#     faces = face_cascade_db. \
#         detectMultiScale(img, 1.1, 19)
#     for (x, y, w, h) in faces:
#         cv.rectangle(img_gray, (x, y),
#                       (x + w, y + h), (0, 255, 0), 2)
#         face_rects.append((x, y, w, h))
#
#     return face_rects


winName = 'IVT Tracker demo'
drawnBox = np.zeros(4)
initialBox = np.zeros(4)
mousexy = (0, 0)
mousedown = False
mouseupdown = False
initialize = False

if __name__ == "__main__":
    print("[INFO] Program started")

    # Parse command line params
    parser = argparse.ArgumentParser(description='Incremental visual tracker.')
    parser.add_argument('-i', '--input', metavar='Input', type=str, help='input file')
    parser.add_argument('-d', '--debug', metavar='Debug', type=int, help='do show debug', default = 0)
    parser.add_argument('-r', '--record', metavar='Record', type=int, help='do record', default = 0)
    parser.add_argument('-t', '--test', metavar='Test', type=int, help='do test', default = 0)
    args = parser.parse_args()

    File = None
    if args.test == 1:
        try:
            np.random.seed(0) # <- for testing only
            File = open("tests/matlab-data.txt", "r") # <- for testing
        except:
            print("[INFO] Skip tests")

    # Create tracker
    tracker = IncrementalTracker(
        affsig = AFFSIG, 
        nparticles = NPARTICLES,
        condenssig = CONDENSSIG, 
        forgetting = FORGETTING, 
        batchsize = BATCHSIZE, 
        tmplShape = (TMPLSIZE, TMPLSIZE), 
        maxbasis = MAXBASIS, 
        errfunc = 'robust' # 'L2'
    )

    # Init cv-window
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    # cv.setMouseCallback(winName, on_mouse, 0)

    # get first frame
    capture = cv.VideoCapture(args.input)
    if args.test == 1: # <- for testing
        for _ in range(299):
            capture.read()

    writer = None
    ret, frame0 = capture.read()
    cv.imwrite(f'originals/img_{0}.jpg', frame0)

    frame0 = cv.resize(frame0, (0, 0), None, RESIZE_RATE, RESIZE_RATE)
    cv.resizeWindow(winName, frame0.shape[1], frame0.shape[0])

    frameNum = 0
    Error = 0.0
    while ret and capture.isOpened():
        if frameNum > 0:
            ret, frame = capture.read()
            cv.imwrite(f'originals/img_{frameNum}.jpg', frame)
            if not ret:
                break
            frame = cv.resize(frame, (0, 0), None, RESIZE_RATE, RESIZE_RATE)
        else:
            frame = frame0

        if not ret:
            print("[INFO] Video ended")
            break
        
        if INITIAL_BOX is None: # manually set initial bounding box
            while frameNum == 0 and not initialize:
                drawImg = frame.copy()
                faces = face_recognise(drawImg)
                face1 = faces[0]
                cv.putText(drawImg, "Draw box around target object", (10, 20), cv.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))
                initialBox[0] = face1[0]
                initialBox[1] = face1[1]
                initialBox[2] = face1[0] + face1[2]
                initialBox[3] = face1[1] + face1[3]

                mousexy = (initialBox[2], initialBox[3])

                cv.rectangle(drawImg,
                        (int(initialBox[0]), int(initialBox[1])),
                        (int(initialBox[2]), int(initialBox[3])),
                        [0,0,255], 2)
                # тупо рисует линии по мышке
                # cv.line(drawImg, (0, int(mousexy[1])), (drawImg.shape[1], int(mousexy[1])), (255, 255, 255), 2)
                # cv.line(drawImg, (int(mousexy[0]), 0), (int(mousexy[0]), drawImg.shape[0]), (255, 255, 255), 2)
                cv.imshow(winName, drawImg)
                cv.waitKey(1)
                # потом вырезать эту переменную
                initialize = True

        # -------------------- CORE -------------------- #
        startTime = time.time()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray) * 0.003921569

        # do tracking
        if frameNum == 0:
            if INITIAL_BOX is None: # <- debug
                w = initialBox[2] - initialBox[0]
                h = initialBox[3] - initialBox[1]
                cx = initialBox[0] + int(w/2)
                cy = initialBox[1] + int(h/2)
                box = np.array([cx, cy, w, h], dtype = np.float32)
                print("Initial box (x, y, w, h): ({0}, {1}, {2}, {3})"
                    .format((box[0] - box[2]/2) / frame.shape[1], (box[1] - box[3]/2) / frame.shape[0], box[2] / frame.shape[1], box[3] / frame.shape[0]))
            else:
                box = INITIAL_BOX
                box[0] = int(box[0] * frame.shape[1])
                box[1] = int(box[1] * frame.shape[0])
                box[2] = int(box[2] * frame.shape[1])
                box[3] = int(box[3] * frame.shape[0])
            est = tracker.track(gray, box)
        else:
            est = tracker.track(gray)

        tmpl = tracker.getTemplate()
        param = tracker.getParam()

        endTime = (time.time() - startTime) * 1000
        # -------------------- //// -------------------- #

        # --------------------- VIZ -------------------- #
        if args.debug:
            debugFrame = makeDetailedFrame(frameNum, frame, tmpl, param, (TMPLSIZE, TMPLSIZE), endTime)
            cv.resizeWindow(winName, debugFrame.shape[1], debugFrame.shape[0])
            cv.imshow(winName, debugFrame)
            if writer is None and args.record:
                writer = cv.VideoWriter('output/output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, (debugFrame.shape[1], debugFrame.shape[0]))
            if args.record:
                writer.write(debugFrame)
        else:
            pred_box = drawEstimatedRect(frame, param, (TMPLSIZE, TMPLSIZE))
            cv.resizeWindow(winName, frame.shape[1], frame.shape[0])
            cv.imshow(winName, frame)
            if writer is None and args.record:
                writer = cv.VideoWriter('output/output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, (frame.shape[1], frame.shape[0]))
            if args.record:
                writer.write(frame)

        if File is not None:
            true = [ float(val) for val in File.readline().split() ]
            Error += np.linalg.norm(param['est'] - np.array(true))
            print(Error)
        # -------------------- //// -------------------- #

        frameNum += 1
        key = cv.waitKey(15)
        if key & 0xFF == ord('q') or key == 27:
            break
    
    if File is not None:
        File.close()
    if writer is not None:
        writer.release()
    capture.release()
    cv.destroyAllWindows()
    print(f"frameNum={frameNum}")
    print("[INFO] Program successfully finished")