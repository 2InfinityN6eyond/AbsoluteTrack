import numpy as np
import cv2

import mediapipe

import sys
sys.path.append('../..')

from lib.common.camera import Fisheye62CameraModel

IMG_WIDTH = 640
IMG_HEIGHT = 480

left_to_right_r = np.array([
    9.9997658245714527e-01, 5.5910744958795095e-04, 6.8206990981942916e-03,
    -5.4903304536865717e-04, 9.9999875583076248e-01, -1.4788169738349651e-03,
    -6.8215174296769373e-03, 1.4750375543776898e-03, 9.9997564528550886e-01
]).reshape(3, 3)

left_to_right_t = np.array([
    -5.9457914254177978e-02, -6.8318101539255457e-05, -1.8101725187729225e-04
])

# k1, k2, k3, k4, p1, p2, k5, k6
distortion_coeffs_left = (
    -3.7539305827469560e-02, 
    -8.7553205432575471e-03,
    2.2015408171895236e-03, 
    -6.6218076061138698e-04,
    0, 0, 0, 0
)
camera_to_world_xf_left = np.eye(4)
rotation_left = np.array([
    [9.9997658245714527e-01,  5.5910744958795095e-04,  6.8206990981942916e-03,],
    [-5.4903304536865717e-04, 9.9999875583076248e-01, -1.4788169738349651e-03,],
    [-6.8215174296769373e-03, 1.4750375543776898e-03,  9.9997564528550886e-01 ],
]).reshape(3, 3)
camera_to_world_xf_left[:3, :3] = rotation_left
#camera_to_world_xf_left[:3, 3] = [
cam_left = Fisheye62CameraModel(
    width   = IMG_WIDTH,
    height  = IMG_HEIGHT,
    f       = (2.3877057700850656e+02, 2.3903223316525276e+02),
    c       = (3.1846939219741773e+02, 2.4685137381795201e+02),
    distort_coeffs = distortion_coeffs_left,
    camera_to_world_xf = np.eye(4)
)

distortion_coeffs_right = (
    -3.6790400486095221e-02, 
    -8.2041573433038941e-03,
    1.0552974220937024e-03, 
    -2.5841665172692902e-04,
    0, 0, 0, 0
)
camera_to_world_xf_right = np.eye(4)
rotation_right = np.array([
    [9.9999470555416226e-01, 1.1490100298631428e-03, 3.0444440536135159e-03,],
    [-1.1535052313709361e-03, 9.9999824663038117e-01, 1.4751819698614872e-03,],
    [-3.0427437166985561e-03, -1.4786859417328980e-03, 9.9999427758290704e-01 ],
]).reshape(3, 3)
camera_to_world_xf_right[:3, :3] = rotation_right
camera_to_world_xf_right[:3, 3] = left_to_right_t
#camera_to_world_xf_right[:3, 3] = [
cam_right = Fisheye62CameraModel(
    width   = IMG_WIDTH,
    height  = IMG_HEIGHT,
    f       = (2.3952183485043457e+02, 2.3981379751051574e+02),
    c       = (3.1286224145189811e+02, 2.5158397962108106e+02),
    distort_coeffs = distortion_coeffs_right,
    camera_to_world_xf = camera_to_world_xf_right
)

def open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
    for CAM_ID in range(-1, CAM_ID_MAX) :
        cap = cv2.VideoCapture(CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if cap.isOpened() :
            print(f"Camera ID {CAM_ID} Frame Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            return cap


if __name__ == "__main__" :
    CAM_ID_MAX = 10
    cap = open_stereo_camera(IMG_WIDTH, IMG_HEIGHT, CAM_ID_MAX)


    MAX_NUM_HANDS = 2
    MODEL_COMPLEXITY = 1
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

    mp_hands = mediapipe.solutions.hands
    mp_hand_left = mp_hands.Hands(
        max_num_hands       = MAX_NUM_HANDS,
        model_complexity    = MODEL_COMPLEXITY,
        min_detection_confidence= MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence = MIN_TRACKING_CONFIDENCE
    )
    mp_hand_right = mp_hands.Hands(
        max_num_hands       = MAX_NUM_HANDS,
        model_complexity    = MODEL_COMPLEXITY,
        min_detection_confidence= MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence = MIN_TRACKING_CONFIDENCE
    )


    p1_list = []
    p2_list = []
    # draw lines from center to boundary.
    image_center = np.array([IMG_WIDTH / 2, IMG_HEIGHT / 2])
    angles = np.linspace(0, np.pi, 50)
    for angle in angles :
        
        half_line_length = min(
            (IMG_WIDTH  / 2) / max(np.cos(angle), 1e-6),
            (IMG_HEIGHT / 2) / max(np.sin(angle), 1e-6)                   #np.abs(np.arccos(angle) * IMG_WIDTH / 2),
            #np.abs(np.arcsin(angle) * IMG_HEIGHT / 2)
        )
        point = np.array([
            np.cos(angle) * half_line_length,
            np.sin(angle) * half_line_length
        ])
        p1 = image_center - point
        p2 = image_center + point
        p1_list.append((int(p1[0]), int(p1[1])))
        p2_list.append((int(p2[0]), int(p2[1])))

    while True :
        ret, frame = cap.read()
        if not ret :
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_left = frame[:, :IMG_WIDTH]
        frame_right = frame[:, IMG_WIDTH:]
        
        results_left = mp_hand_left.process(frame_left)
        results_right = mp_hand_right.process(frame_right)
        
        mp_hand_pose_dict_left = {}
        mp_hand_pose_dict_right = {}
        
        if results_left.multi_handedness :
            for handedness, landmark in zip(
                results_left.multi_handedness,
                results_left.multi_hand_landmarks
            ) :
                hand_index = handedness.classification[0].index
                hand_pose = np.array(list(map(
                    lambda l : [l.x, l.y, l.z],
                    landmark.landmark
                ))) * np.array([
                    IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH
                ])
                mp_hand_pose_dict_left[hand_index] = hand_pose
        
        if results_right.multi_handedness :
            for handedness, landmark in zip(
                results_right.multi_handedness,
                results_right.multi_hand_landmarks
            ) :
                hand_index = handedness.classification[0].index
                hand_pose = np.array(list(map(
                    lambda l : [l.x, l.y, l.z],
                    landmark.landmark
                ))) * np.array([
                    IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH
                ])
                mp_hand_pose_dict_right[hand_index] = hand_pose
        
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

        # plot original window pose
        for hand_index, hand_pose in mp_hand_pose_dict_left.items() :
            for point in hand_pose :
                x, y, z = point
                cv2.circle(frame_left, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        for hand_index, hand_pose in mp_hand_pose_dict_right.items() :
            for point in hand_pose :
                x, y, z = point
                cv2.circle(frame_right, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        # plot reprojected without distortion
        # for hand_index, keypoints_window_orig in mp_hand_pose_dict_left.items() :
        #     keypoints_eye_orig = cam_left.window_to_eye(
        #         keypoints_window_orig[:, :2]
        #     )
        #     keypoints_window_reproj = cam_left.eye_to_window_undistorted(
        #         keypoints_eye_orig
        #     )
        #     for point in keypoints_window_reproj :
        #         x, y = point
        #         cv2.circle(frame_left, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        # for hand_index, keypoints_window_orig in mp_hand_pose_dict_right.items() :
        #     keypoints_eye_orig = cam_right.window_to_eye(
        #         keypoints_window_orig[:, :2]
        #     )
        #     keypoints_window_reproj = cam_right.eye_to_window_undistorted(
        #         keypoints_eye_orig
        #     )
        #     for point in keypoints_window_reproj :
        #         x, y = point
        #         cv2.circle(frame_right, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        # plot reprojected and distorted window pose
        for hand_index, keypoints_window_orig in mp_hand_pose_dict_left.items() :
            keypoints_eye_orig = cam_left.window_to_eye(
                keypoints_window_orig[:, :2]
            )
            keypoints_window_reproj = cam_left.eye_to_window(
                keypoints_eye_orig
            )
            for point in keypoints_window_reproj :
                x, y = point
                cv2.circle(frame_left, (int(x), int(y)), 1, (255, 0, 0), -1)
        
        for hand_index, keypoints_window_orig in mp_hand_pose_dict_right.items() :
            keypoints_eye_orig = cam_right.window_to_eye(
                keypoints_window_orig[:, :2]
            )
            keypoints_window_reproj = cam_right.eye_to_window(
                keypoints_eye_orig
            )
            for point in keypoints_window_reproj :
                x, y = point
                cv2.circle(frame_right, (int(x), int(y)), 1, (255, 0, 0), -1)
        
        
        
        # draw lines from center to boundary.
        for p1, p2 in zip(p1_list, p2_list) :
            cv2.line(frame_left, p1, p2, (0, 0, 0), 1)
            cv2.line(frame_right, p1, p2, (0, 0, 0), 1)
        
        cv2.imshow('frame_left', frame_left)
        cv2.imshow('frame_right', frame_right)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()