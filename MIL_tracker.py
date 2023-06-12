import cv2
from original_annotation import AnnotationReader
import sys
import time

# tracker = cv2.legacy.TrackerCSRT_create()
# tracker = cv2.TrackerCSRT_create()
tracker = cv2.TrackerMIL_create()

NAME = "video_2023-04-05_22-42-06"
DATASET_PATH = f'dataset/{NAME}/'
SAVE = True
OUTPUT_NAME = 'output/no_fair.avi'
root_mean_temp = 0
root_mean_time = 0
root_mean_iou = 0
time_all = 0

reader = AnnotationReader(f'{DATASET_PATH}new.json', DATASET_PATH)


ground_boxes = reader.get_boxes(0)
ground_boxes[2] = ground_boxes[2] - ground_boxes[0]
ground_boxes[3] = ground_boxes[3] - ground_boxes[1]

frame = reader.get_image(0)
if SAVE:
    writer = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                            (frame.shape[1], frame.shape[0]))

tracker_init = tracker.init(frame, ground_boxes)


for i in range(reader.get_size()):
    frame = reader.get_image(i)
    ground_boxes = reader.get_boxes(i)

    start_time = time.time_ns()
    ok, bbox = tracker.update(frame)
    end_time = time.time_ns()
    time_all += end_time - start_time
    root_mean_time += 1

    cv2.rectangle(frame, (ground_boxes[0], ground_boxes[1]), (ground_boxes[2], ground_boxes[3]), (255, 0, 0),
                 2)
    # test print coordinates of predicted bounding box for all frames
    if ok == True:

        (x, y, w, h) = [int(v) for v in bbox]
        # use predicted bounding box coordinates to draw a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
        # проверяю таким образом, отвернулся или нет
        area_ground_boxes = reader.get_area(ground_boxes)
        pred_boxes = (x, y, x + w, y + h)
        iou = reader.get_iou(ground_boxes, pred_boxes)
        root_mean_iou += iou
        root_mean_temp += 1

        cv2.putText(frame, f"{iou}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('Single Track', frame)
    if SAVE:
        writer.write(frame)

    # press 'q' to break loop and close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('avg_iou', root_mean_iou / root_mean_temp)
print('time_avg', time_all / root_mean_time)