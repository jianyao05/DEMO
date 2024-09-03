import cv2
import mediapipe as mp
import numpy as np


def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):

    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)


    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle = 0, startAngle = -90, endAngle = -180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle = 0, startAngle = 0, endAngle = -90, color = box_color, thickness = -1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle = 0, startAngle = 90, endAngle = 180, color = box_color, thickness = -1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle = 0, startAngle = 0, endAngle = 90, color = box_color, thickness = -1)

    return img




def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i+pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame


'''
Parameters:
p1: A NumPy array representing the first point in 2D space (e.g., coordinates of a landmark).
p2: A NumPy array representing the second point in 2D space.
ref_pt: A NumPy array representing the reference point from which the angle is measured. 
By default, it is set to the origin (0, 0).

Functionality:

Reference Vectors: The function calculates the vectors p1_ref and p2_ref by subtracting the ref_pt from p1 and p2, 
respectively. This transforms the points into a local coordinate system where the reference point is at the origin.

Cosine of the Angle: It computes the cosine of the angle between the two vectors using the dot product formula. 
The cosine value is clipped between -1.0 and 1.0 to avoid numerical errors from floating-point calculations.

Angle Calculation: The angle in radians is found using the inverse cosine (np.arccos), and then it is converted 
to degrees.

Return Value: The angle is returned as an integer in degrees.
'''
def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degree = int(180 / np.pi) * theta

    return int(degree)

'''
extract the normalized coordinates of a landmark from the MediaPipe pose landmark results and convert them into pixel 
coordinates based on the dimensions of the frame (image).

Parameters:
pose_landmark: This is the output from MediaPipe, which contains landmarks detected in the image.
key: This specifies which landmark you want to extract (e.g., 'nose', 'left_shoulder', etc.).
frame_width: The width of the image frame in pixels.
frame_height: The height of the image frame in pixels.

Functionality:
It multiplies the normalized x and y coordinates of the specified landmark by the width and height of the frame, respectively, to convert them into pixel coordinates.
It returns a NumPy array containing the denormalized coordinates.
'''
def get_landmark_array(pose_landmark, key, frame_width, frame_height):

    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])


def draw_text(
        img,
        msg,
        width=8,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
        box_offset=(20, 10),
):
    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))

    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)

    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return text_size


def draw_angle_on_image(img, point1, point2, ref_point):
    angle = find_angle(point1, point2, ref_point)
    text_position = (point2[0] + 10, point2[1])  # Position to draw the angle text
    draw_text(img, f'Angle: {angle}°', pos=text_position)
    return img

def get_mediapipe_pose(
                        static_image_mode = False,
                        model_complexity = 1,
                        smooth_landmarks = True,
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5

                      ):
    pose = mp.solutions.pose.Pose(
                                    static_image_mode = static_image_mode,
                                    model_complexity = model_complexity,
                                    smooth_landmarks = smooth_landmarks,
                                    min_detection_confidence = min_detection_confidence,
                                    min_tracking_confidence = min_tracking_confidence
                                 )
    return pose


# Main function to run video capture
def main():
    # Initialize MediaPipe Pose
    pose = get_mediapipe_pose()
    mp_draw = mp.solutions.drawing_utils

    # Open video capture (0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # Convert the BGR frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get pose landmarks
        results = pose.process(img_rgb)

        # Draw landmarks and calculate angles if landmarks are detected
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Example: Get the coordinates for angle calculation
            left_shoulder = get_landmark_array(results.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                                               frame.shape[1], frame.shape[0])
            left_elbow = get_landmark_array(results.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                                            frame.shape[1], frame.shape[0])
            left_wrist = get_landmark_array(results.pose_landmarks, mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                                            frame.shape[1], frame.shape[0])

            # Calculate angle at the left elbow
            angle = find_angle(left_shoulder, left_elbow, left_wrist)

            # Draw angle on the image
            draw_angle_on_image(frame, left_shoulder, left_elbow, left_wrist)

        # Show the output frame
        cv2.imshow("Pose Estimation", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()