import multiprocessing as mp
import numpy as np
import os
import json
import time
import socket


import cv2

import sys
sys.path.append('..')

import lib.common.camera as camera
from lib.tracker.tracker import HandTracker
from lib.common.camera import Fisheye62CameraModel

from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
from lib.common.hand import HandModel
from lib.tracker.perspective_crop import landmarks_from_hand_pose

class UmeTracker(mp.Process):
    def __init__(
        self,
        config,
        shared_array_mono,
        shared_mp_pose_array,
        shared_ume_pose_array,
        mp2ume,
        ume2imgviz,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.shared_array_mono = shared_array_mono
        self.shared_mp_pose_array = shared_mp_pose_array
        self.shared_ume_pose_array = shared_ume_pose_array
        self.mp2ume = mp2ume
        self.ume2imgviz = ume2imgviz
        self.stop_event = stop_event
        self.verbose = verbose
        
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
        self.cam_left = Fisheye62CameraModel(
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
        self.cam_right = Fisheye62CameraModel(
            width   = IMG_WIDTH,
            height  = IMG_HEIGHT,
            f       = (2.3952183485043457e+02, 2.3981379751051574e+02),
            c       = (3.1286224145189811e+02, 2.5158397962108106e+02),
            distort_coeffs = distortion_coeffs_right,
            camera_to_world_xf = camera_to_world_xf_right
        )

    def run(self):
        # initialize mono shared array
        self.shared_array_mono_list = [
            np.frombuffer(
                self.shared_array_mono[i].buf, dtype=np.uint8
            ).reshape((
                2 if self.config.is_stereo else 1, 
                self.config.camera.image_height, 
                self.config.camera.image_width,
            )) for i in range(self.config.buffer.size)
        ]
        
        # initialize mp pose shared array
        self.shared_mp_pose_list = [
            np.frombuffer(
                self.shared_mp_pose_array[i].buf, dtype=np.float32
            ).reshape((
                2 if self.config.is_stereo else 1,
                self.config.media_pipe.max_num_hands,
                self.config.media_pipe.num_keypoints,
                3
            )) for i in range(self.config.buffer.size)
        ]
        
        # initialize ume pose shared array
        self.shared_ume_pose_list = [
            np.frombuffer(
                self.shared_ume_pose_array[i].buf, dtype=np.float32
            ).reshape((
                self.config.ume_tracker.max_num_hands,
                self.config.ume_tracker.num_keypoints,
                3
            )) for i in range(self.config.buffer.size)
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
        tracker = HandTracker(model, HandTrackerOpts())
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        serverAddressPort = ("127.0.0.1", 5052)

        fps_1 = 0
        fps_2 = 0
        while not self.stop_event.is_set():
            stt_1 = time.time()
            
            if self.mp2ume.empty():
                continue
            
            index, mp_hand_pose_dict = self.mp2ume.get()
            
            stt_2 = time.time()
            
            frame = self.shared_array_mono_list[index]
            # mp_pose_results = self.shared_mp_pose_list[index]
            
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
            
            # hand_pose_window_left_cam = {}
            # for hand_idx in range(2):
            #     if mp_pose_results[0, hand_idx].mean() > 0.1:
            #         hand_pose_window_left_cam[hand_idx] = mp_pose_results[0, hand_idx]
            
            # hand_pose_window_right_cam = {}
            # for hand_idx in range(2):
            #     if mp_pose_results[1, hand_idx].mean() > 0.1:
            #         hand_pose_window_right_cam[hand_idx] = mp_pose_results[1, hand_idx]
            
            hand_pose_window_left_cam = mp_hand_pose_dict[0]
            hand_pose_window_right_cam = mp_hand_pose_dict[1]
            
            # print(f"left cam {list(hand_pose_window_left_cam.keys())}, {list(hand_pose_window_right_cam.keys())}")
            
            crop_camera_dict = tracker.gen_crop_cameras_from_stereo_camera_with_window_hand_pose(
                camera_left = self.cam_left,
                camera_right = self.cam_right,
                window_hand_pose_left = hand_pose_window_left_cam,
                window_hand_pose_right = hand_pose_window_right_cam
            )
            
            # if len(crop_camera_dict.keys()):
            #     print(
            #         [
            #             (list(crop_camera_dict[hand_idx].keys()), hand_idx) for hand_idx in crop_camera_dict.keys()
            #         ]
            #     )


            '''
            res = tracker.track_frame_analysis(
                fisheye_stereo_input_frame, 
                hand_model, 
                crop_camera_dict,
                None
            )
            '''
            tracked_keypoints_dict = {}
            
            
            
            '''
            self.shared_ume_pose_list[index][:] = 0
            for hand_idx in res.hand_poses.keys() :
                tracked_keypoints = landmarks_from_hand_pose(
                    hand_model, res.hand_poses[hand_idx], hand_idx
                )
                self.shared_ume_pose_list[index][hand_idx] = tracked_keypoints
                tracked_keypoints_dict[hand_idx] = tracked_keypoints
            '''
            
            # if 1 in tracked_keypoints_dict:
            #     print(tracked_keypoints_dict[1].mean(axis=0).astype(np.int32))
                            
            
            projected_keypoints_dict = {0:{}, 1:{}}
            
            '''
            for cam_idx in range(2 if self.config.is_stereo else 1):
                per_cam_projected_keypoints_dict = {}
                for hand_idx in tracked_keypoints_dict.keys():
                    tracked_keypoints = tracked_keypoints_dict[hand_idx]
                    projected_keypoints = fisheye_stereo_input_frame.views[cam_idx].camera.world_to_eye(
                        tracked_keypoints
                    )
                    per_cam_projected_keypoints_dict[hand_idx] = projected_keypoints
                projected_keypoints_dict[cam_idx] = per_cam_projected_keypoints_dict
            '''
            
            
            
            fps_2 = 0.8 * fps_2 + 0.2 * (1 / (time.time() - stt_2))
            fps_1 = 0.8 * fps_1 + 0.2 * (1 / (time.time() - stt_1))
            
            if self.verbose:
                print(f"                                   UME FPS: {int(fps_1)}, {int(fps_2)}, {len(hand_pose_window_left_cam)}, {len(hand_pose_window_right_cam)}")
            
            
            self.ume2imgviz.put((
                index,
                mp_hand_pose_dict,
                tracked_keypoints_dict,
                projected_keypoints_dict
            ))