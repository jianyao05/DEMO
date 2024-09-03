import cv2
import numpy as np
import time
import pose_module as pm
import winsound


duration = 1000  # milliseconds
freq = 440  # Hz


cap = cv2.VideoCapture(0)
# img = cv2.imread('datasets/squatting.jpg')

detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        # left lunges
        angle = detector.findAngle(img, 23, 25, 27)
        # right lunges
        # angle = detector.findAngle(img, 24, 26, 28)
        # left bicep
        # angle = detector.findAngle(img, 11, 13, 15)
        # right bicep
        # angle = detector.findAngle(img, 12, 14, 16)

        per = np.interp(angle, (75, 175), (100, 0))
        bar = np.interp(angle, (75, 175), (100, 650))
        print(angle, per)

        # checking if the exercise is one FULL REP
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
                winsound.Beep(freq, duration)
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        print(count)
        # DRAWING OF BAR
        cv2.rectangle(img, (1110, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # DRAW COUNTER
        cv2.rectangle(img, (0, 550), (150, 850), (0, 255,0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 678), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    # fps
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows