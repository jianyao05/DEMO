import cv2
import numpy as np
import time
import weird1 as pm



cap = cv2.VideoCapture(0)

detector = pm.poseDetector()

while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        angle = detector.Angle(img, 11, 13)
        current_state = detector.get_state(angle)
        print(current_state)
        detector.update_state_sequence(current_state)
        print(detector.update_state_sequence(current_state))
        proper_raise , improper_raise = detector.counter(img, current_state)
        cv2.putText(img, str(int(proper_raise)), (45, 678), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        print(lmList[13][2])

    cv2.imshow('Image', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows