import os
import json
import time
import cv2
import mediapipe as mp
import numpy as np
import socket

import sys
sys.path.append('..')

from lib.common.camera import Fisheye62CameraModel
from lib.common.hand import HandModel
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, ViewData, InputFrame
from lib.models.model_loader import load_pretrained_model

def open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
    for CAM_ID in range(-1, CAM_ID_MAX) :
        cap = cv2.VideoCapture(CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if cap.isOpened() :
            print(f"Camera ID {CAM_ID} Frame Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            return cap

# initialize camera model
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

MP_CONNECTION_MAP = [
    (0, 1), (1, 2), (2, 3), (3, 4), # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17), # palm
]
HAND_CONNECTION_MAP = [
    [5, 6], [6, 7], [7, 0], # thumb
    [5, 8], [8, 9], [9, 10], [10, 1], # index
    [5, 11], [11, 12], [12, 13], [13, 2], # middle
    [5, 14], [14, 15], [15, 16], [16, 3], # ring
    [5, 17], [17, 18], [18, 19], [19, 4], # pinky
    [8, 11], [11, 14], [14, 17]
]


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

IMG_SETTING_OPTIONS = {
    "0": (1920, 1080),
    "1": (1280, 720),
    "2": (640, 480),
}
DEFAULT_IMAGE_SETTING = "2"



'''
# set cv2 window position
cv2.namedWindow('cam0', cv2.WINDOW_NORMAL)
cv2.moveWindow('cam0', 0, 0)

cv2.namedWindow('cam1', cv2.WINDOW_NORMAL)
cv2.moveWindow('cam1', 700, 0)

cv2.namedWindow('cam 0 hand 0', cv2.WINDOW_NORMAL)
cv2.moveWindow('cam 0 hand 0', 0, 500)

cv2.namedWindow('cam 1 hand 0', cv2.WINDOW_NORMAL)
cv2.moveWindow('cam 1 hand 0', 300, 500)

cv2.namedWindow('cam 0 hand 1', cv2.WINDOW_NORMAL)
cv2.moveWindow('cam 0 hand 1', 0, 1000)

cv2.namedWindow('cam 1 hand 1', cv2.WINDOW_NORMAL)
cv2.moveWindow('cam 1 hand 1', 300, 1000)
'''

if __name__ == "__main__":
    print('start')
    IMAGE_SETTING = input(
        f"Enter the image setting (0: {IMG_SETTING_OPTIONS['0']}, 1: {IMG_SETTING_OPTIONS['1']}, 2: {IMG_SETTING_OPTIONS['2']}): "
    ) or DEFAULT_IMAGE_SETTING
    IMAGE_WIDTH, IMAGE_HEIGHT = IMG_SETTING_OPTIONS[IMAGE_SETTING]
    
    CAM_ID_MAX = 10
    cap = open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # PRETRAIN_PATH = "/home/hjp/HJP/KUAICV/Hand/AbsoluteTrack/agora/hand_landmarker.task"

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_left = mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    )
    hands_right = mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    )

    
    UMETRACK_ROOT = ".."
    HAND_MODEL_DATA_PATH = os.path.join(UMETRACK_ROOT, "dataset/generic_hand_model.json")
    with open(HAND_MODEL_DATA_PATH, 'r') as f:
        data = json.load(f)
    hand_model = HandModel.from_json(data)
    
    model_name = "pretrained_weights.torch"
    model_path = os.path.join(UMETRACK_ROOT, "pretrained_models", model_name)
    model = load_pretrained_model(model_path)
    model.eval()
    tracker_opts = HandTrackerOpts()
    # tracker_opts.hand_ratio_in_crop = 0.5
    tracker = HandTracker(model, tracker_opts)

    # draw hand pose
    mp_handedness_color_map = {
        0: (0, 0, 255), # RED
        1: (0, 255, 0), # GREEN
    }
    ume_handedness_color_map = {
        0: (10, 10, 55), # RED
        1: (10, 55, 10), # GREEN
    }
    

    fps_outer = 0
    fps_inner = 0
    while cap.isOpened():
        stt = time.time()
        ret, frame_stereo = cap.read()
        if not ret:
            break
        
        frame_left = frame_stereo[:, :IMAGE_WIDTH]
        frame_right = frame_stereo[:, IMAGE_WIDTH:]
        

        frame_left_mono = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frame_right_mono = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # process mediapipe
        result_left = hands_left.process(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
        result_right = hands_right.process(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))
        
        window_hand_pose_left = {}
        if result_left.multi_handedness is not None:
            for handedness, landmark in zip(
                result_left.multi_handedness,
                result_left.multi_hand_landmarks
            ) :
                hand_index = 1 - handedness.classification[0].index
                hand_pose = np.array(list(map(
                    lambda l : [l.x, l.y, l.z],
                    landmark.landmark
                ))) * np.array([
                    frame_width,
                    frame_height,
                    frame_width
                ])
                window_hand_pose_left[hand_index] = hand_pose
                
        window_hand_pose_right = {}
        if result_right.multi_handedness is not None:
            for handedness, landmark in zip(
                result_right.multi_handedness,
                result_right.multi_hand_landmarks
            ) :
                hand_index = 1 - handedness.classification[0].index
                hand_pose = np.array(list(map(
                    lambda l : [l.x, l.y, l.z],
                    landmark.landmark
                ))) * np.array([
                    frame_width,
                    frame_height,
                    frame_width
                ])
                window_hand_pose_right[hand_index] = hand_pose
    
        
        fisheye_stereo_input_frame = InputFrame(
            views = [
                ViewData(
                    image = frame_left_mono,
                    camera = cam_left,
                    camera_angle = 0,
                ),
                ViewData(
                    image = frame_right_mono,
                    camera = cam_right,
                    camera_angle = 0,
                )
            ]
        )


        crop_camera_dict = tracker.gen_crop_cameras_from_stereo_camera_with_window_hand_pose(
            camera_left = cam_left,
            camera_right = cam_right,
            window_hand_pose_left = window_hand_pose_left,
            window_hand_pose_right = window_hand_pose_right
        )


        res = tracker.track_frame_analysis(
            fisheye_stereo_input_frame, 
            hand_model, 
            crop_camera_dict,
            None
        )
        # res = tracker.track_frame(    
        #     fisheye_stereo_input_frame, 
        #     hand_model, 
        #     crop_camera_dict,
        # )
        
        tracked_keypoints_dict = {}
        for hand_idx in res.hand_poses.keys() :
            tracked_keypoints = landmarks_from_hand_pose(
                hand_model, res.hand_poses[hand_idx], hand_idx
            )
            tracked_keypoints_dict[hand_idx] = tracked_keypoints

        if 0 in tracked_keypoints_dict and 1 in tracked_keypoints_dict :
            # print(
            #     tracked_keypoints_dict[0].mean(axis=0).astype(np.int32),
            #     tracked_keypoints_dict[1].mean(axis=0).astype(np.int32),
            # )
            
            content = []
            for hand_idx in tracked_keypoints_dict.keys() :
                data = tracked_keypoints_dict[hand_idx].copy()
                data[:, :2] *= -1
                FLIP_X = True
                if FLIP_X :
                    data[:, 0] *= -1
                content.append(str(data.flatten().astype(int).tolist()))
            
            content = ";".join(content)
            
            sock.sendto(str.encode(str(content)), serverAddressPort)
        
        projected_keypoints_dict = {}
        for cam_idx in range(len(fisheye_stereo_input_frame.views)):
            camera = fisheye_stereo_input_frame.views[cam_idx].camera
            per_cam_projected_keypoints_dict = {}
            for hand_idx in tracked_keypoints_dict.keys():
                tracked_keypoints = tracked_keypoints_dict[hand_idx]
                projected_keypoints = camera.eye_to_window(
                    camera.world_to_eye(tracked_keypoints)
                )
                per_cam_projected_keypoints_dict[hand_idx] = projected_keypoints
            projected_keypoints_dict[cam_idx] = per_cam_projected_keypoints_dict

        
        fps_inner = 0.5 * fps_inner + 0.5 * (1 / (time.time() - stt))

        # plot mediapipe hand pose
        for hand_index, hand_pose in window_hand_pose_left.items():
            for point in hand_pose:
                x, y, z = point
                cv2.circle(frame_left, (int(x), int(y)), 2, mp_handedness_color_map[hand_index], -1)
        for con in MP_CONNECTION_MAP :
            for hand_idx, hand_pose in window_hand_pose_left.items() :
                cv2.line(
                    frame_left, 
                    hand_pose[con[0]][:2].astype(np.int32), 
                    hand_pose[con[1]][:2].astype(np.int32), 
                    mp_handedness_color_map[hand_idx],
                    1
                )


        for hand_index, hand_pose in window_hand_pose_right.items():
            for point in hand_pose:
                x, y, z = point
                cv2.circle(frame_right, (int(x), int(y)), 2, mp_handedness_color_map[hand_index], -1)
        for con in MP_CONNECTION_MAP :
            for hand_idx, hand_pose in window_hand_pose_right.items() :
                cv2.line(
                    frame_right, 
                    hand_pose[con[0]][:2].astype(np.int32), 
                    hand_pose[con[1]][:2].astype(np.int32), 
                    mp_handedness_color_map[hand_idx],
                    1
                )

        # plot uume hand pose 
        for hand_index, hand_pose in projected_keypoints_dict[0].items():
            for point in hand_pose:
                x, y = point
                cv2.circle(frame_left, (int(x), int(y)), 2, ume_handedness_color_map[hand_index], -1)
        for con in HAND_CONNECTION_MAP :
            for hand_idx, hand_pose in projected_keypoints_dict[0].items() :
                cv2.line(
                    frame_left, 
                    hand_pose[con[0]][:2].astype(np.int32), 
                    hand_pose[con[1]][:2].astype(np.int32), 
                    ume_handedness_color_map[hand_idx],
                    2
                )

        for hand_index, hand_pose in projected_keypoints_dict[1].items():
            for point in hand_pose:
                x, y = point
                cv2.circle(frame_right, (int(x), int(y)), 2, ume_handedness_color_map[hand_index], -1)
        for con in HAND_CONNECTION_MAP :
            for hand_idx, hand_pose in projected_keypoints_dict[1].items() :
                cv2.line(
                    frame_right, 
                    hand_pose[con[0]][:2].astype(np.int32), 
                    hand_pose[con[1]][:2].astype(np.int32), 
                    ume_handedness_color_map[hand_idx],
                    2
                )



        cv2.imshow('cam0', frame_left)
        cv2.imshow('cam1', frame_right)
    
        fps_outer = 0.5 * fps_outer + 0.5 * (1 / (time.time() - stt))
        # print(f"FPS: {fps_outer:.2f}, {fps_inner:.2f}")
    
        k = cv2.waitKey(1)
        if k==49:
            break
        
    cap.release()
    cv2.destroyAll_Windows()