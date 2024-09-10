import cv2
import numpy as np
import time
import woah as pm

pic = 'sideway_man.jpg'

cap = cv2.VideoCapture(0)

detector = pm.poseDetector()

while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        shld_z = lmList[11][3]
        wrist_z = lmList[15][3]


        cv2.putText(img, str(lmList[11][3]), (45, 678), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.putText(img, str(lmList[15][3]), (45, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)


    cv2.imshow('Image', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows