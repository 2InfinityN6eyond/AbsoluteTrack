import os
import json
import socket
import time
import tyro
import numpy as np
import cv2
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import shared_memory


import sys
sys.path.append('..')


from lib.common.camera import Fisheye62CameraModel
from lib.common.hand import HandModel
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, ViewData, InputFrame
from lib.models.model_loader import load_pretrained_model

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



class UmeTracker(mp.Process):
    def __init__(
        self,
        config,
        shm_name_list,
        mp2ume,
        ume2vz,
        stop_event,
        verbose = False
    ):

        super().__init__()
        self.config = config
        self.shm_name_list = shm_name_list
        self.mp2ume = mp2ume
        self.ume2vz = ume2vz
        self.stop_event = stop_event
        self.verbose = verbose
        self.cam_left = None
        self.cam_right = None
        
    def run(self):
        try:
            print('ume-run', flush=True)
        except Exception as e:
            print(f'Error in ume-tracker run method: {e}')

        self.cam_left = Fisheye62CameraModel(
            width   = IMG_WIDTH,
            height  = IMG_HEIGHT,
            f       = (2.3877057700850656e+02, 2.3903223316525276e+02),
            c       = (3.1846939219741773e+02, 2.4685137381795201e+02),
            distort_coeffs = distortion_coeffs_left,
            camera_to_world_xf = np.eye(4)
        )
        self.cam_right = Fisheye62CameraModel(
            width   = IMG_WIDTH,
            height  = IMG_HEIGHT,
            f       = (2.3952183485043457e+02, 2.3981379751051574e+02),
            c       = (3.1286224145189811e+02, 2.5158397962108106e+02),
            distort_coeffs = distortion_coeffs_right,
            camera_to_world_xf = camera_to_world_xf_right
        )

        self.shm_list = [
            shared_memory.SharedMemory(
                name = shm_name
            ) for shm_name in self.shm_name_list
        ]
        self.frames_array_list = [
            np.ndarray(
                shape = (
                    self.config.camera.n_views,
                    self.config.camera.image_height,
                    self.config.camera.image_width,
                ),
                dtype = np.uint8,
                buffer = shm.buf
            ) for shm in self.shm_list
        ]
        
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

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        serverAddressPort = ("127.0.0.1", 5052)

        fps = 0
        while not self.stop_event.is_set():
            stt = time.time()
            
            if self.mp2ume.empty():
                continue

            index, mp_hand_pose_dict = self.mp2ume.get()

            frame = self.frames_array_list[index].copy()


            fisheye_stereo_input_frame = InputFrame(
                views = [
                    ViewData(
                        image = frame[0].copy(),
                        camera = self.cam_left,
                        camera_angle = 0,
                    ),
                    ViewData(
                        image = frame[1].copy(),
                        camera = self.cam_right,
                        camera_angle = 0,
                    )
                ]
            )

            hand_pose_window_left_cam = mp_hand_pose_dict[0]
            hand_pose_window_right_cam = mp_hand_pose_dict[1]

            crop_camera_dict = tracker.gen_crop_cameras_from_stereo_camera_with_window_hand_pose(
                camera_left = self.cam_left,
                camera_right = self.cam_right,
                window_hand_pose_left = hand_pose_window_left_cam,
                window_hand_pose_right = hand_pose_window_right_cam
            )

            # res = tracker.track_frame_analysis(
            #     fisheye_stereo_input_frame, 
            #     hand_model, 
            #     crop_camera_dict,
            #     None
            # )

            res = tracker.track_frame(
                fisheye_stereo_input_frame,
                hand_model,
                crop_camera_dict,
            )
            
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



            projected_keypoints_dict = {0:{}, 1:{}}
            
            for cam_idx in range(2):
                camera = fisheye_stereo_input_frame.views[cam_idx].camera
                per_cam_projected_keypoints_dict = {}
                for hand_idx in tracked_keypoints_dict.keys():
                    tracked_keypoints = tracked_keypoints_dict[hand_idx]
                    projected_keypoints = camera.eye_to_window(
                        camera.world_to_eye(tracked_keypoints)
                    )
                    per_cam_projected_keypoints_dict[hand_idx] = projected_keypoints
                projected_keypoints_dict[cam_idx] = per_cam_projected_keypoints_dict            
            
            self.ume2vz.put((
                index,
                mp_hand_pose_dict,
                tracked_keypoints_dict,
                projected_keypoints_dict
            ))

