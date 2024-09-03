import cv2
import mediapipe as mp
import numpy as np

def draw_dotted_line(self, img, lm_coord, start, end, line_color):
        pix_step = 0

        for i in range(start, end + 1, 8):
            cv2.circle(img, (lm_coord, i + pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

        return img

def get_landmark_array(pose_landmark, key, img_width, img_height):

    denorm_x = int(pose_landmark[key].x * img_width)
    denorm_y = int(pose_landmark[key].y * img_height)

    return np.array([denorm_x, denorm_y])

def get_landmark_features(kp_results, dict_features, leftORright, img_width, img_height):

    if leftORright == 'nose':
        return get_landmark_array(kp_results, dict_features[leftORright], img_width, img_height)

    elif leftORright == 'left' or leftORright == 'right':
        shoulder_coord = get_landmark_array(kp_results, dict_features[leftORright]['shoulder'], img_width, img_height)
        elbow_coord   = get_landmark_array(kp_results, dict_features[leftORright]['elbow'], img_width, img_height)
        wrist_coord   = get_landmark_array(kp_results, dict_features[leftORright]['wrist'], img_width, img_height)
        hip_coord   = get_landmark_array(kp_results, dict_features[leftORright]['hip'], img_width, img_height)
        knee_coord   = get_landmark_array(kp_results, dict_features[leftORright]['knee'], img_width, img_height)
        ankle_coord   = get_landmark_array(kp_results, dict_features[leftORright]['ankle'], img_width, img_height)
        foot_coord   = get_landmark_array(kp_results, dict_features[leftORright]['foot'], img_width, img_height)

        return shoulder_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord

    else:
        raise ValueError("feature needs to be either 'nose', 'left' or 'right")



    def process(self, img):

        height, width, _ = img.shape
        keypoints = self.pose.process(img)
        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)
