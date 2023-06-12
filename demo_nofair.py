
from norfair import Detection, Tracker, Video, draw_tracked_objects
import cv2 as cv
import time
import numpy as np
from original_annotation import AnnotationReader

NAME = 'video_2023-04-05_22-41-53'
DATASET_PATH = f'dataset/{NAME}/'

reader = AnnotationReader(f'{DATASET_PATH}/new.json', DATASET_PATH)

video = Video(input_path=f"input/{NAME}.mov")
tracker = Tracker(distance_function="euclidean", distance_threshold=100)
initialBox = np.zeros(4)

frameNum = 0
root_mean_temp = 0
root_mean_time = 0
root_mean_iou = 0
time_all = 0
SAVE = True
OUTPUT_NAME = 'output/no_fair.avi'

frame = np.array(reader.get_image(0))
frame = reader.get_image(0)
writer = cv.VideoWriter(OUTPUT_NAME, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                        (frame.shape[1], frame.shape[0]))

for frame in video:
   try:
      ground_boxes = reader.get_boxes(frameNum)
      cv.rectangle(frame, (ground_boxes[0], ground_boxes[1]), (ground_boxes[2], ground_boxes[3]), (255, 0, 0),
                    2)
      print()
      # face = face_recognise(frame)

      initialBox[0] = ground_boxes[0]
      initialBox[1] = ground_boxes[1]
      initialBox[2] = ground_boxes[2]
      initialBox[3] = ground_boxes[3]

      # detections = np.array(face)
      detections = np.array([ground_boxes])



      norfair_detections = [Detection(np.array(points)) for points in detections]
      time_start = time.time_ns()
      tracked_objects = tracker.update(detections=norfair_detections)
      time_stop = time.time_ns()
      time_all += time_stop - time_start
      root_mean_time += 1

      # проверяю таким образом, отвернулся или нет
      area_ground_boxes = reader.get_area(ground_boxes)
      if area_ground_boxes > 30:
         if len(tracked_objects) == 0:
            iou = 0
         else:
            pred_boxes = tracked_objects[0].estimate[0]
            iou = reader.get_iou(ground_boxes, pred_boxes)

            cv.rectangle(frame, (int(pred_boxes[0]), int(pred_boxes[1])), (int(pred_boxes[2]), int(pred_boxes[3])), (0, 0, 255),
                         2)

         root_mean_iou += iou
         root_mean_temp += 1

         cv.putText(frame, f"{iou}", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


      draw_tracked_objects(frame, tracked_objects)



   except Exception:
      pass

   video.write(frame)
   cv.imshow('nofair', frame)
   if SAVE:
      writer.write(frame)

   frameNum += 1

print('avg_iou', root_mean_iou / root_mean_temp)
print('time_avg', time_all / root_mean_time)