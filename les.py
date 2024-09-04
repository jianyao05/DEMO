import cv2
import numpy as np
import time
import pose_module as pm



cap = cv2.VideoCapture(0)

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
        reference = lmList[13][1:]
        angle = detector.findAngle(11, 13, reference)
        x, y = lmList[11][1:]
        print(angle)
        multiplier = -1
        cv2.ellipse(img, np.array(lmList[11][1:]), (30, 30),
                    angle=0, startAngle=90, endAngle=90 + multiplier * angle,
                    color=(255,255,255), thickness=3,)
        detector.draw_dotted_line(img, np.array(lmList[11][1:]), start=(lmList[11][2]) - 40, end=(lmList[11][2]) + 40,
                         line_color=(0, 0, 255))

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