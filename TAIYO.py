import cv2
import numpy as np
import time
import RYO as pm

# ------------------- INITIALISES WEBCAM --------------
cap = cv2.VideoCapture(0)

detector = pm.poseDetector()

while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, True)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        pass

    cv2.imshow('Image', img)
    cv2.waitKey(1)

    # ------------------- EXIT PROGRAM VIA 'Q' BUTTON --------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows