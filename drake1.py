import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(point1, point2, point3):
    # Calculate vectors
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)

    # Dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Angle in radians
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    # Recolor the frame to RGB (required for MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make Pose Detection
    results = pose.process(frame_rgb)

    # Extract landmarks if present
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get the coordinates of the shoulder, elbow, and wrist
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]

        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]

        # Create a point directly below the shoulder by adjusting the y-coordinate
        point_below_shoulder = [shoulder[0], shoulder[1] + 0.2, shoulder[2]]  # Add 0.2 to y to move below the shoulder

        # Calculate the angle using the shoulder as the pivot
        angle = calculate_angle(point_below_shoulder, shoulder, elbow)

        # Display the angle on the frame
        cv2.putText(frame, str(int(angle)),
                    (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        print(shoulder)

    # Display the frame
    cv2.imshow('Angle Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
