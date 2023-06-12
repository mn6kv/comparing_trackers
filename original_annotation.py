import json
import cv2 as cv
import numpy as np


class AnnotationReader:
    def __init__(self, jsonpath="dsc_train/new.json", image_path='dsc_train/'):
        self._image_path = image_path
        with open(jsonpath) as file:
            self._json_file = json.load(file)

    def get_boxes(self, num):
        """
        Returns
        """
        boxes = self._json_file[num]['annotations'][0]['coordinates']
        x_center = int(boxes['x'])
        y_center = int(boxes['y'])
        width = int(boxes['width'])
        height = int(boxes['height'])

        x1 = int((x_center - int(width / 2)))
        x2 = int((x_center + int(width / 2)))
        y1 = int((y_center - int(height / 2)))
        y2 = int((y_center + int(height / 2)))

        return [x1, y1, x2, y2]

    def get_image(self, frame_num: int):
        filename = self._json_file[frame_num]['image']
        img = cv.imread(self._image_path+filename)
        return img

    def get_size(self):
        return len(self._json_file)

    def get_shape(self):
        "return x shape, y shape"
        return self.get_image(0).shape[1], self.get_image(0).shape[0]

    def get_name(self, num):
        "return x shape, y shape"
        return self._json_file[num]['image']

    @staticmethod
    def is_cross(a, b):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]  # прямоугольник А
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]  # прямоугольник B
        # это были координаты точек диагонали по каждому прямоугольнику

        # 1. Проверить условия перекрытия, например, если XПA<XЛB ,
        #    то прямоугольники не пересекаются,и общая площадь равна нулю.
        #   (это случай, когда они справа и слева) и аналогично, если они сверху
        #    и снизу относительно друг друга.
        #    (XПА - это  Х Правой точки прямоугольника А)
        #    (ХЛВ - Х Левой точки прямоугольника В )
        #    нарисуй картинку (должно стать понятнее)

        xA = [ax1, ax2]  # координаты x обеих точек прямоугольника А
        xB = [bx1, bx2]  # координаты x обеих точке прямоугольника В

        yA = [ay1, ay2]  # координаты x обеих точек прямоугольника А
        yB = [by1, by2]  # координаты x обеих точек прямоугольника В

        if max(xA) < min(xB) or max(yA) < min(yB) or min(yA) > max(yB) or max(xB) < min(xA):
            return False  # не пересекаются

        # 2. Определить стороны прямоугольника образованного пересечением,
        # например,
        # если XПA>XЛB, а XЛA<XЛB, то ΔX=XПA−XЛB

        elif max(xA) > min(xB) and min(xA) < min(xB):
            dx = max(xA) - min(xB)
            return True  # пересекаются
        else:
            return True  # пересекаются
    @staticmethod
    def get_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        if not AnnotationReader.is_cross(boxA, boxB):
            return 0
        # compute the area of intersection rectangle
        interArea = abs(max((abs(xB - xA), 0)) * max(abs(yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        boxAArea = abs(boxAArea)
        boxBArea = abs(boxBArea)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        iou = abs(iou)
        if iou > 1:
            return 0
        # return the intersection over union value
        return abs(iou)


    @staticmethod
    def get_area(boxA):
        xA = boxA[0]
        yA = boxA[1]
        xB = boxA[2]
        yB = boxA[3]
        width = xA - xB
        height = yB - yA
        return abs(width * height)
