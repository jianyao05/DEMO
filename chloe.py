import cv2
import mediapipe as mp
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

    def draw_dotted_line(self, img, lm_coord, start, end, line_color):
            pix_step = 0

            for i in range(start, end + 1, 8):
                cv2.circle(img, (lm_coord, i + pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

            return img




    def findPose(self, img, draw=True):
        ### Draws and form the Skeletal system
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    def findPosition(self, img, draw=True):
        ### Finds position [0, 250, 400] eg. nose [keypoint, x, y]
        ###  lmList = detector.findPosition(img, False) include in the final code
        ### Draws only the Key points to make it clearer eg. Elbows
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                img_height, img_width, c = img.shape
                cx, cy = int(lm.x * img_width), int(lm.y * img_height)
                self.lmList.append([id, cx, cy]) #id of feature, x-pixel, y-pixel
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList


    def









