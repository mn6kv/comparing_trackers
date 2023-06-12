import os
import numpy as np
import cv2 as cv
import argparse
from itertools import islice
from src.tracker import *
from src.model_specific import *
import json
import cv2 as cv
import time
from original_annotation import AnnotationReader
if not os.path.exists('output'):
    os.makedirs('output')

MODEL_PATH = r'res/lbpcascade_frontalface.xml'


winName = 'IVT Tracker demo'
initialBox = np.zeros(4)
mousexy = (0, 0)
initialize = False

NAME = 'video_2023-04-05_22-42-06'
DATASET_PATH = f'dataset/{NAME}/'
SAVE = True
OUTPUT_NAME = 'output/original.avi'

reader = AnnotationReader(f'{DATASET_PATH}new.json', DATASET_PATH)

frameNum = 0
root_mean_iou_temp = 0
root_mean_time_temp = 0
root_mean_iou = 0

if __name__ == "__main__":
    print("[INFO] Program started")

    # Parse command line params
    # parser = argparse.ArgumentParser(description='Incremental visual tracker.')
    # parser.add_argument('-i', '--input', metavar='Input', type=str, help='input file')
    # parser.add_argument('-d', '--debug', metavar='Debug', type=int, help='do show debug', default=0)
    # parser.add_argument('-r', '--record', metavar='Record', type=int, help='do record', default=0)
    # parser.add_argument('-t', '--test', metavar='Test', type=int, help='do test', default=0)
    # args = parser.parse_args()
    #
    # File = None
    # if args.test == 1:
    #     try:
    #         np.random.seed(0)  # <- for testing only
    #         File = open("tests/matlab-data.txt", "r")  # <- for testing
    #     except:
    #         print("[INFO] Skip tests")

    # Create tracker
    tracker = IncrementalTracker(
        affsig=AFFSIG,
        nparticles=NPARTICLES,
        condenssig=CONDENSSIG,
        forgetting=FORGETTING,
        batchsize=BATCHSIZE,
        tmplShape=(TMPLSIZE, TMPLSIZE),
        maxbasis=MAXBASIS,
        errfunc='robust'  # 'L2'
    )

    # Init cv-window
    frame0 = reader.get_image(0)
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    # cv.setMouseCallback(winName, on_mouse, 0)

    # frame0 = cv.resize(frame0, (0, 0), None, RESIZE_RATE, RESIZE_RATE)
    cv.resizeWindow(winName, frame0.shape[1], frame0.shape[0])

    writer = None

    Error = 0.0

    frameNum = 0

    middle_square = 0

    all_time = 0
    first_time_detect = True
    for frameNum in range(reader.get_size()):

        if frameNum > 0:
            # ret, frame = capture.read()
            frame = reader.get_image(frameNum)
            # find_contours_etalon(frame.copy())
            # frame = cv.resize(frame, (0, 0), None, RESIZE_RATE, RESIZE_RATE)
        else:
            frame = frame0

        # cv.imshow(winName, frame)

        if first_time_detect:  # manually set initial bounding box
            # while frameNum == 0 and not initialize:
            ground_boxes = reader.get_boxes(frameNum)
            drawImg = frame.copy()



            cv.putText(drawImg, "Draw box around target object", (10, 20), cv.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))
            initialBox[0] = ground_boxes[0]
            initialBox[1] = ground_boxes[1]
            initialBox[2] = ground_boxes[2]
            initialBox[3] = ground_boxes[3]

            mousexy = (initialBox[2], initialBox[3])

            cv.rectangle(drawImg,
                         (int(initialBox[0]), int(initialBox[1])),
                         (int(initialBox[2]), int(initialBox[3])),
                         [0, 255, 0], 2)
            # тупо рисует линии по мышке
            # cv.line(drawImg, (0, int(mousexy[1])), (drawImg.shape[1], int(mousexy[1])), (255, 255, 255), 2)
            # cv.line(drawImg, (int(mousexy[0]), 0), (int(mousexy[0]), drawImg.shape[0]), (255, 255, 255), 2)
            # cv.imshow(winName, drawImg)
            cv.waitKey(1)
            # потом вырезать эту переменную
            initialize = True
        if not initialize:
            continue
        # -------------------- CORE -------------------- #

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray) * 0.003921569

        # do tracking
        if first_time_detect:
            if INITIAL_BOX is None:  # <- debug
                w = initialBox[2] - initialBox[0]
                h = initialBox[3] - initialBox[1]
                cx = initialBox[0] + int(w / 2)
                cy = initialBox[1] + int(h / 2)
                box = np.array([cx, cy, w, h], dtype=np.float32)
                print("Initial box (x, y, w, h): ({0}, {1}, {2}, {3})"
                      .format((box[0] - box[2] / 2) / frame.shape[1], (box[1] - box[3] / 2) / frame.shape[0],
                              box[2] / frame.shape[1], box[3] / frame.shape[0]))
            else:
                box = INITIAL_BOX
                box[0] = int(box[0])
                box[1] = int(box[1])
                box[2] = int(box[2])
                box[3] = int(box[3])

                # box[0] = int(box[0] * frame.shape[1])
                # box[1] = int(box[1] * frame.shape[0])
                # box[2] = int(box[2] * frame.shape[1])
                # box[3] = int(box[3] * frame.shape[0])
            startTime = time.time_ns()

            est = tracker.track(gray, box)
            endTime = (time.time_ns() - startTime)
            all_time += endTime
            root_mean_time_temp += 1
            first_time_detect = False
        else:
            try:
                startTime = time.time_ns()
                est = tracker.track(gray)
                endTime = (time.time_ns() - startTime)
                root_mean_time_temp += 1
                all_time += endTime

            except Exception:
                pass

        tmpl = tracker.getTemplate()
        param = tracker.getParam()
        boxes = param['est']

        # -------------------- //// -------------------- #

    # --------------------- VIZ -------------------- #
        pred_box = drawEstimatedRect(frame, param, (TMPLSIZE, TMPLSIZE))
        max_pred_x = sorted(pred_box, key=lambda x: x[0] )[0][0]
        min_pred_x = sorted(pred_box, key=lambda x: x[0] )[-1][0]
        min_pred_y = sorted(pred_box, key=lambda x: x[1])[0][1]
        max_pred_y = sorted(pred_box, key=lambda x: x[1])[-1][1]

        cv.resizeWindow(winName, frame.shape[1], frame.shape[0])

        ground_boxes = reader.get_boxes(frameNum)
        # x1 = int((x_center - int(width / 2)  )* RESIZE_RATE)
        # x2 = int((x_center + int(width / 2 ) )* RESIZE_RATE)
        # y1 = int((y_center - int(height / 2 ))  * RESIZE_RATE)
        # y2 = int((y_center + int(height / 2 )) * RESIZE_RATE)
        cv.rectangle(frame, (ground_boxes[0], ground_boxes[1]), (ground_boxes[2], ground_boxes[3]), (255, 0, 0), 2)

        if reader.get_area(ground_boxes) > 30:
            iou = reader.get_iou(ground_boxes, [max_pred_x, max_pred_y, min_pred_x, min_pred_y])
            # iou = reader.bb_intersection_over_union(ground_boxes, [max_pred_x, max_pred_y, min_pred_x, min_pred_y])
            root_mean_iou += iou
            root_mean_iou_temp += 1

            cv.putText(frame, f"{iou}", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv.imshow(winName, frame)


        if writer is None and SAVE:
            writer = cv.VideoWriter(OUTPUT_NAME, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                    (frame.shape[1], frame.shape[0]))
        if SAVE:
            writer.write(frame)

        # if File is not None:
        #     true = [float(val) for val in File.readline().split()]
        #     Error += np.linalg.norm(param['est'] - np.array(true))
        #     print(Error)
        # -------------------- //// -------------------- #

        key = cv.waitKey(15)
        if key & 0xFF == ord('q') or key == 27:
            break
        if first_time_detect:
            first_time_detect = False
        # while True:
        #     key = cv.waitKey(15)
        #     if key & 0xFF == ord('q') or key == 27:
        #         break
    # if File is not None:
    #     File.close()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()
    print(frameNum)
    print('iou = ', root_mean_iou / root_mean_iou_temp)
    print("[INFO] Program successfully finished")
    print('avg_time=', int(all_time / root_mean_time_temp))