import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
import cv2
import dlib
from imutils import face_utils
import imutils
import time
import numpy as np
import argparse
from collections import deque
from math import hypot
import pyglet


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_PLAIN


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = ver_line_lenght / hor_line_lenght
    return ratio


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    return left_eye_region


class MyApp(QDialog):
    def __init__(self):
        super(MyApp, self).__init__()
        loadUi(r'C:\Users\Administrator\PycharmProjects\pythonProject3\venv\qttest.ui', self)
        self.m = 0
        self.k = 0
        self.gazer = 0
        self.gazel = 0
        self.COUNTER = 0
        self.t = 0
        self.start_webcam()
        self.image = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            r'C:\Users\Administrator\PycharmProjects\pythonProject3\venv\shape_predictor_68_face_landmarks.dat')

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def update_frame(self):
        _, self.image = self.capture.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rects = self.detector(self.gray, 0)  # detect face in the current frame (var = image)

        for rect in self.rects:
            landmarks = self.predictor(self.gray, rect)

            # Detect blinking
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio < 0.275:
                cv2.putText(self.image, "BLINKING", (50, 150), font, 6, (255, 0, 0))
                self.COUNTER += 1
            else:
                if self.COUNTER >= 3:
                    self.t += 1
                self.COUNTER = 0

            cv2.putText(self.image, str(self.t), (50, 200), font, 6, (255, 0, 0))
            # Gaze detection
            height, width, _ = self.image.shape
            mask = np.zeros((height, width), np.uint8)
            # +++++++++
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            cv2.polylines(mask, [gaze_ratio_right_eye], True, 255, 2)
            cv2.fillPoly(mask, [gaze_ratio_right_eye], 255)
            eye = cv2.bitwise_and(self.gray, self.gray, mask=mask)

            min_x = np.min(gaze_ratio_right_eye[:, 0])
            max_x = np.max(gaze_ratio_right_eye[:, 0])
            min_y = np.min(gaze_ratio_right_eye[:, 1])
            max_y = np.max(gaze_ratio_right_eye[:, 1])

            gray_eye = eye[min_y: max_y, min_x: max_x]
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            height, width = threshold_eye.shape
            left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
            left_side_white = cv2.countNonZero(left_side_threshold)

            right_side_threshold = threshold_eye[0: height, int(width / 2): width]
            right_side_white = cv2.countNonZero(right_side_threshold)

            if left_side_white == 0:
                gaze_right = 1
            elif right_side_white == 0:
                gaze_right = 5
            else:
                gaze_right = left_side_white / right_side_white
            # ********
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            cv2.polylines(mask, [gaze_ratio_left_eye], True, 255, 2)
            cv2.fillPoly(mask, [gaze_ratio_left_eye], 255)
            eye1 = cv2.bitwise_and(self.gray, self.gray, mask=mask)

            min_x1 = np.min(gaze_ratio_left_eye[:, 0])
            max_x1 = np.max(gaze_ratio_left_eye[:, 0])
            min_y1 = np.min(gaze_ratio_left_eye[:, 1])
            max_y1 = np.max(gaze_ratio_left_eye[:, 1])

            gray_eye1 = eye1[min_y1: max_y1, min_x1: max_x1]
            _, threshold_eye1 = cv2.threshold(gray_eye1, 70, 255, cv2.THRESH_BINARY)
            height, width = threshold_eye1.shape
            left_side_threshold1 = threshold_eye1[0: height, 0: int(width / 2)]
            left_side_white1 = cv2.countNonZero(left_side_threshold1)

            right_side_threshold1 = threshold_eye1[0: height, int(width / 2): width]
            right_side_white1 = cv2.countNonZero(right_side_threshold1)

            if left_side_white1 == 0:
                gaze_left = 1
            elif right_side_white1 == 0:
                gaze_left = 5
            else:
                gaze_left = left_side_white1 / right_side_white1

            gaze_ratio = (gaze_right + gaze_left) / 2
            # print(gaze_ratio_left_eye)
            cv2.putText(self.image, str(gaze_ratio), (100, 50), font, 2, (0, 0, 255), 3)
            cv2.polylines(self.image, [gaze_ratio_left_eye], True, (0, 0, 255), 2)
            cv2.polylines(self.image, [gaze_ratio_right_eye], True, (0, 0, 255), 2)

            if gaze_ratio < 0.7:
                cv2.putText(self.image, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                self.gazer += 1
            else:
                if self.gazer >= 3:
                    self.k += 1
                self.gazer = 0
            if 0.7 <= gaze_ratio <= 1.7:
                cv2.putText(self.image, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)

            if gaze_ratio > 1.7:
                cv2.putText(self.image, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                self.gazel += 1
            else:
                if self.gazel >= 3:
                    self.k -= 1
                self.gazel = 0

            if self.k == 1:
                self.on1.setStyleSheet("background-color: rgb(255, 255, 0)")
                self.off2.setStyleSheet(" ")
                self.on2.setStyleSheet(" ")
                self.off1.setStyleSheet(" ")
                if self.t == 1:
                    self.status1.setText('ON')
                self.t = 0
            elif self.k == 2:
                self.off1.setStyleSheet("background-color: rgb(255, 255, 0)")
                self.on1.setStyleSheet(" ")
                self.on2.setStyleSheet(" ")
                self.off2.setStyleSheet(" ")
                if self.t == 1:
                    self.status1.setText('OFF')
                self.t = 0
            elif self.k == 3:
                self.on2.setStyleSheet("background-color: rgb(255, 255, 0)")
                self.on1.setStyleSheet(" ")
                self.off2.setStyleSheet(" ")
                self.off1.setStyleSheet(" ")
                if self.t == 1:
                    self.status2.setText('ON')
                self.t = 0
            elif self.k == 4:
                self.off2.setStyleSheet("background-color: rgb(255, 255, 0)")
                self.on1.setStyleSheet(" ")
                self.on2.setStyleSheet(" ")
                self.off1.setStyleSheet(" ")
                if self.t == 1:
                    self.status2.setText('OFF')
                self.t = 0
            if self.k > 4:
                self.k = 4
            if self.k < 1:
                self.k = 1

            for n in range(0, 17):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(self.image, (x, y), 2, (0, 255, 0), -1)
            self.displayImage(self.image, 1)

    def stop_webcam(self):
        time.sleep(5)

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows [1]=col [2]=channel
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.window.setPixmap(QPixmap.fromImage(outImage))
            self.window.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyApp()
    window.setWindowTitle('Control sys. via eye detection')
    window.show()
    sys.exit(app.exec_())
