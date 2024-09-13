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

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        shoulder = lmList[11][1:]
        elbow = lmList[13][1:]
        point_below_shoulder = [shoulder[0], shoulder[1] + 0.2, shoulder[2]]  # Add 0.2 to y to move below the shoulder

        angle = detector.calculate_angle(point_below_shoulder, shoulder, elbow)
        print(angle)

        # CONNECTING THE KEY JOINTS REGARDING ANGLES
        cv2.line(img, shoulder[0:2], elbow[0:2], (102, 204, 255), 4, cv2.LINE_AA)
        cv2.putText(img, str(int(angle)),
                    (int(elbow[0]), int(elbow[1])),  # Position the text at the elbow joint
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

    # ------------------- EXIT PROGRAM VIA 'Q' BUTTON --------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows


