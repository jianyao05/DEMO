import cv2
import mediapipe as mp
import time
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

    def findPosition(self, img, draw=False):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def find_angle(self, p1, p2, ref_pt=np.array([0, 0])):
        p1_ref = p1 - ref_pt
        p2_ref = p2 - ref_pt

        cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        degree = int(180 / np.pi) * theta

        return int(degree)


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            # Calculate the elbow angle
            shoulder = np.array([lmList[11][1], lmList[11][2]])  # Shoulder
            elbow = np.array([lmList[12][1], lmList[12][2]])  # Elbow
            wrist = np.array([lmList[14][1], lmList[14][2]])  # Wrist

            # Calculate angle at elbow
            elbow_angle = detector.find_angle(elbow - shoulder, wrist - elbow)

            # Display elbow angle
            cv2.putText(img, f'Elbow Angle: {elbow_angle}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img, str(int(fps)), (78, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
