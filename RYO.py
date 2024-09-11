import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=1, smooth_landmarks=self.smooth,
                                     enable_segmentation=False, smooth_segmentation=True,
                                     min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        # ------------------- STORES LANDMARK AND COORDINATES IN LIST eg. [id, x, y, z] --------------
        self.list = []


        # ------------------- COUNTER --------------
        self.SQUAT_COUNT = 0
        self.IMPROPER_SQUAT = 0

        self.INCORRECT_POSTURE = False

    # ------------------- FINDS AND DRAW LANDMARKS --------------
    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    # ------------------- STORES LANDMARK AND COORDINATES IN LIST eg. [id, x, y, z] --------------
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def draw_dotted_line(self, img, lm_coord, start, end, line_color, direction='vertical'):
        # If the direction is vertical, draw the line vertically, otherwise horizontally
        pix_step = 0

        if direction == 'vertical':
            # Draw vertical dotted line
            for i in range(start, end + 1, 8):
                cv2.circle(img, (lm_coord[0], i + pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)
        elif direction == 'horizontal':
            # Draw horizontal dotted line
            for i in range(start, end + 1, 8):
                cv2.circle(img, (i + pix_step, lm_coord[1]), 2, line_color, -1, lineType=cv2.LINE_AA)

        return img