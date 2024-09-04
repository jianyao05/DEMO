'''
Join Landmark# ------------------------------------------------------------


# Join landmarks.
cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
cv2.line(frame, ankle_coord, knee_coord ,self.COLORS['light_blue'], 4,  lineType=self.linetype)
cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)

# Plot landmark points
cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)



current_state = self._get_state(int(knee_vertical_angle))
self.state_tracker['curr_state'] = current_state
self._update_state_sequence(current_state)



# -------------------------------------- COMPUTE COUNTERS --------------------------------------
'''
import cv2
import mediapipe as mp
import numpy as np


'''
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

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def draw_dotted_line(self, img, lm_coord, start, end, line_color):
        pix_step = 0

        for i in range(start, end + 1, 8):
            cv2.circle(img, (lm_coord[0], i + pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

        return img

    def findAngle(self, p1, p2, ref_pt=np.array([0, 0])):
        Point1 = np.array(self.lmList[p1][1:])  # Convert to NumPy array
        Point2 = np.array(self.lmList[p2][1:])  # Convert to NumPy array

        p1_ref = Point1 - ref_pt
        p2_ref = Point2 - ref_pt

    def findAngle(self, p1, p2, ref_pt=np.array([0, 0])):
        # Convert to NumPy arrays for easier calculations
        Point1 = np.array(self.lmList[p1][1:])  # Get coordinates of point 1
        Point2 = np.array(self.lmList[p2][1:])  # Get coordinates of point 2

        delta_y = Point2[1] - Point1[1]  # Difference in y-coordinates
        delta_x = Point2[0] - Point1[0]  # Difference in x-coordinates

        # Get angle in radians
        angle_rad = np.arctan2(delta_y, delta_x)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360

        return int(angle_deg)


'''



reference = lmList[13][1:]
    angle = detector.findAngle(11, 13, reference)
    x, y = lmList[11][1:]
    print(angle)
    multiplier = -1
    cv2.ellipse(img, np.array(lmList[11][1:]), (30, 30),
                angle=0, startAngle=90, endAngle=90 + multiplier * angle,
                color=(255, 255, 255), thickness=3, )
    detector.draw_dotted_line(img, np.array(lmList[11][1:]), start=(lmList[11][2]) - 40, end=(lmList[11][2]) + 40,
                              line_color=(0, 0, 255))
    knee_text_coord_x = lmList[11][1] + 45
    cv2.putText(img, str(int(angle)), (knee_text_coord_x, lmList[11][2] + 45), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 255, 255), 2)

    ### joinin of landmarks
    cv2.line(img, lmList[23][1:], lmList[11][1:], (102, 204, 255), 4, cv2.LINE_AA)
    cv2.line(img, lmList[11][1:], lmList[13][1:], (102, 204, 255), 4, cv2.LINE_AA)
    cv2.circle(img, lmList[11][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, lmList[13][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, lmList[23][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)

    if angle > 100:
        yuhangsuckballs += 1
        print('amt of reps', yuhangsuckballs)



