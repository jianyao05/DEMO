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

        self.list = []
        self.INCORRECT_POSTURE = False
        self.SQUAT_COUNT = 0
        self.IMPROPER_SQUAT = 0

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


    def Angle(self, img, third_connection, feature1, feature2):
        reference = self.lmList[feature2][1:]

        point1 = np.array(self.lmList[feature1][1:])  # Convert to NumPy array
        point2 = np.array([self.lmList[feature2][1], 0])  # Convert to NumPy array
        p1_ref = point1 - reference
        p2_ref = point2 - reference
        cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        degree = int(180 / np.pi) * theta

        ###

        multiplier = -1
        cv2.ellipse(img, np.array(self.lmList[11][1:]), (30, 30),
                    angle=0, startAngle=90, endAngle=90 + multiplier * degree,
                    color=(255, 255, 255), thickness=3, )
        self.draw_dotted_line(img, np.array(self.lmList[11][1:]), start=(self.lmList[11][2]) - 40, end=(self.lmList[11][2]) + 40,
                              line_color=(0, 0, 255))
        knee_text_coord_x = self.lmList[11][1] + 45
        cv2.putText(img, str(int(degree)), (knee_text_coord_x, self.lmList[11][2] + 45), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 255), 2)


        cv2.line(img, self.lmList[third_connection][1:], self.lmList[feature1][1:], (102, 204, 255), 4, cv2.LINE_AA)
        cv2.line(img, self.lmList[feature1][1:], self.lmList[feature2][1:], (102, 204, 255), 4, cv2.LINE_AA)
        cv2.circle(img, self.lmList[feature1][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, self.lmList[feature2][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, self.lmList[third_connection][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)




        return int(degree)

    def get_state(self, angle):
        state = None

        if 0 <= angle <= 30:
            state = 1
        elif 35 <= angle <= 75:
            state = 2
        elif 80 <= angle <= 100:
            state = 3
        return f"s{state}" if state else None

    def update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.list) and (self.list.count('s2')) == 0) or (('s3' in self.list) and (self.list.count('s2') == 1)):
                self.list.append(state)
                '''If 's3' hasn’t been added yet, only one 's2' can be added.
                   If 's3' has been added, one more 's2' can be added, but only if it has appeared once before.'''
        elif state == 's3':
            if (state not in self.list) and ('s2' in self.list):
                self.list.append(state)
        return self.list

    def counter(self, state):
        if state == 's1':
            if len(self.list) == 3 and not self.INCORRECT_POSTURE:
                self.SQUAT_COUNT += 1

            elif 's2' in self.list and len(self.list) == 1:
                self.IMPROPER_SQUAT += 1


            elif self.INCORRECT_POSTURE:
                self.IMPROPER_SQUAT += 1

            self.list = []
            self.INCORRECT_POSTURE = False
            print('squats',self.SQUAT_COUNT)
            print('improper',self.IMPROPER_SQUAT)







'''    
    angle_of_movement = detector.Angle(img, )   
        
    
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


state_tracker = {'state_seq': []}
    state_tracker['curr_state'] = current_state
    self._update_state_sequence(current_state)

'''
