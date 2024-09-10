import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to calculate angle between two vectors in 3D space
def calculate_3d_angle(point1, point2, point3):
    # Create vectors (elbow->shoulder, elbow->wrist)
    vector_1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
    vector_2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]

    # Calculate dot product and magnitudes
    dot_product = sum(p1 * p2 for p1, p2 in zip(vector_1, vector_2))
    magnitude_1 = math.sqrt(sum(p1 ** 2 for p1 in vector_1))
    magnitude_2 = math.sqrt(sum(p2 ** 2 for p2 in vector_2))

    # Calculate the angle between the vectors
    angle_radians = math.acos(dot_product / (magnitude_1 * magnitude_2))
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the pose landmarks
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Get shoulder, elbow, wrist landmarks in 3D
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # Calculate the 3D angle at the armpit (shoulder-elbow-wrist)
        angle = calculate_3d_angle(shoulder, elbow, wrist)

        # Display the calculated angle on the frame
        cv2.putText(frame, f'Armpit Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks for better visualization
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
