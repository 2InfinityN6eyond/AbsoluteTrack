# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import json
#import av
import cv2
import fnmatch
import pickle
import numpy as np
import torch
from functools import partial
from multiprocessing import Pool
from typing import Optional, Tuple, NamedTuple, Dict, Any, List
import socket
from dataclasses import dataclass

import lib.data_utils.fs as fs
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.common.hand import NUM_HANDS, NUM_LANDMARKS_PER_HAND
from lib.common.camera import CameraModel, read_camera_from_json
from lib.tracker.tracking_result import SingleHandPose
from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker, HandTrackerOpts, InputFrame, ViewData
# from lib.tracker.video_pose_data import SyncedImagePoseStream

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


HAND_CONNECTION_MAP = [
    [5, 6], [6, 7], [7, 0], # thumb
    [5, 8], [8, 9], [9, 10], [10, 1], # index
    [5, 11], [11, 12], [12, 13], [13, 2], # middle
    [5, 14], [14, 15], [15, 16], [16, 3], # ring
    [5, 17], [17, 18], [18, 19], [19, 4], # pinky
    [8, 11], [11, 14], [14, 17]
]

# draw hand pose
gt_handedness_color_map = {
    0: (0, 0, 255), # RED
    1: (0, 255, 0), # GREEN
}
ume_handedness_color_map = {
    0: (10, 10, 55), # RED
    1: (10, 55, 10), # GREEN
}
result_handedness_color_map = {
    0: (200, 200, 255), # RED
    1: (200, 255, 200), # GREEN
}

class HandModel(NamedTuple):
    joint_rotation_axes: torch.Tensor
    joint_rest_positions: torch.Tensor
    joint_frame_index: torch.Tensor
    joint_parent: torch.Tensor
    joint_first_child: torch.Tensor
    joint_next_sibling: torch.Tensor
    landmark_rest_positions: torch.Tensor
    landmark_rest_bone_weights: torch.Tensor
    landmark_rest_bone_indices: torch.Tensor
    hand_scale: Optional[torch.Tensor]
    mesh_vertices: Optional[torch.Tensor] = None
    mesh_triangles: Optional[torch.Tensor] = None
    dense_bone_weights: Optional[torch.Tensor] = None
    joint_limits: Optional[torch.Tensor] = None
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'HandModel':
        return cls(**{
            key: torch.tensor(value) if value is not None else None
            for key, value in json_data.items()
        })

    def to_json(self) -> Dict[str, Any]:
        return {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in self._asdict().items()
        }


@dataclass
class HandPoseLabels:
    cameras: List[CameraModel]
    camera_angles: List[float]
    camera_to_world_transforms: np.ndarray
    hand_model: HandModel
    joint_angles: np.ndarray
    wrist_transforms: np.ndarray
    hand_confidences: np.ndarray

    def __len__(self):
        return len(self.joint_angles)
                   

def _load_json(p: str):
    with fs.open(p, "rb") as bf:
        return json.load(bf)

def load_hand_model_from_dict(hand_model_dict) -> HandModel:
    hand_tensor_dict = {}
    for k, v in hand_model_dict.items():
        if isinstance(v, list):
            hand_tensor_dict[k] = torch.Tensor(v)
        else:
            hand_tensor_dict[k] = v

    hand_model = HandModel(**hand_tensor_dict)
    return hand_model


def _load_hand_pose_labels(p: str) -> HandPoseLabels:
    labels = _load_json(p)
    cameras = [read_camera_from_json(c) for c in labels["cameras"]]
    camera_angles = labels["camera_angles"]
    hand_model = load_hand_model_from_dict(labels["hand_model"])
    joint_angles = np.array(labels["joint_angles"])
    wrist_transforms = np.array(labels["wrist_transforms"])
    hand_confidences = np.array(labels["hand_confidences"])
    camera_to_world_transforms = np.array(labels["camera_to_world_transforms"])

    return HandPoseLabels(
        cameras=cameras,
        camera_angles=camera_angles,
        camera_to_world_transforms=camera_to_world_transforms,
        hand_model=hand_model,
        joint_angles=joint_angles,
        wrist_transforms=wrist_transforms,
        hand_confidences=hand_confidences,
    )


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)




def _track_sequence(
    #input_output: Tuple[str, str],
    video_path: str,
    data_path: str,
    model_path: str,
    override: bool = False,
) -> Optional[np.ndarray]:
    #data_path, output_path = input_output
    # if not override and fs.exists(output_path):
    #     logger.info(f"Skipping '{data_path}' since output path '{output_path}' already exists")
    #     return None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)
    
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 4
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_list = []
    stereo_bgr_frame_list = []
    stereo_mono_frame_list = []
    for frame_idx in range(n_frames) :
        ret, frame = cap.read()
        if ret :
            frame_list.append(frame)
            stereo_bgr_frame_list.append([
                frame[:, :frame_width], 
                frame[:, frame_width:frame_width*2],
                frame[:, frame_width*2:frame_width*3],
                frame[:, frame_width*3:]
            ])
            frame_mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stereo_mono_frame_list.append([
                frame_mono[:, :frame_width],
                frame_mono[:, frame_width:frame_width*2],
                frame_mono[:, frame_width*2:frame_width*3],
                frame_mono[:, frame_width*3:]
            ])
        else :
            break

    stereo_bgr_frame_array = np.array(stereo_bgr_frame_list)
    stereo_mono_frame_array = np.array(stereo_mono_frame_list)
    

    data = _load_hand_pose_labels(data_path)
    hand_model = data.hand_model
    
    
    gt_tracking_list = []
    for frame_idx in range(n_frames) :
        gt_tracking = {}
        for hand_idx in range(0, 2) :
            if data.hand_confidences[frame_idx, hand_idx] > 0 :
                gt_tracking[hand_idx] = SingleHandPose(
                    joint_angles=data.joint_angles[frame_idx, hand_idx],
                    wrist_xform=data.wrist_transforms[frame_idx, hand_idx],
                    hand_confidence=data.hand_confidences[frame_idx, hand_idx]
                )
        gt_tracking_list.append(gt_tracking)



    while True :

        # logger.info(f"Processing {data_path}...")
        model = load_pretrained_model(model_path)
        model.eval()

        print("==============================")
        print("image steam opened")

        
        tracked_keypoints = np.zeros([NUM_HANDS, len(stereo_bgr_frame_list), NUM_LANDMARKS_PER_HAND, 3])
        valid_tracking = np.zeros([NUM_HANDS, len(stereo_bgr_frame_list)], dtype=bool)
        tracker = HandTracker(model, HandTrackerOpts())
        
        print("==============================")
        print("tracker initialized")
        for frame_idx in range(n_frames):
            
            views = []
            for cam_idx in range(0, len(data.cameras)):
                cur_camera = data.cameras[cam_idx].copy(
                    camera_to_world_xf=data.camera_to_world_transforms[frame_idx, cam_idx],
                )
                views.append(ViewData(
                    image=stereo_mono_frame_array[frame_idx, cam_idx, :], 
                    camera=cur_camera, 
                    camera_angle=data.camera_angles[cam_idx]
                ))
                
            input_frame = InputFrame(views=views)
            
            
            gt_tracking = gt_tracking_list[frame_idx]
            
            crop_cameras = tracker.gen_crop_cameras(
                [view.camera for view in input_frame.views],
                data.camera_angles,
                hand_model,
                gt_tracking,
                min_num_crops=1,
            )
            res = tracker.track_frame(input_frame, hand_model, crop_cameras)


            tracked_keypoints_dict = {}
            for hand_idx in res.hand_poses.keys():
                tracked_keypoints_dict[hand_idx] = landmarks_from_hand_pose(
                    hand_model, res.hand_poses[hand_idx], hand_idx
                )
                
                
                tracked_keypoints[hand_idx, frame_idx] = landmarks_from_hand_pose(
                    hand_model, res.hand_poses[hand_idx], hand_idx
                )
                
                valid_tracking[hand_idx, frame_idx] = True
                
            if 0 in tracked_keypoints_dict and 1 in tracked_keypoints_dict :
                content = []
                for hand_idx in tracked_keypoints_dict.keys() :
                    tracked_data = tracked_keypoints_dict[hand_idx].copy()
                    tracked_data[:, :2] *= -1
                    FLIP_X = True
                    if FLIP_X :
                        tracked_data[:, 0] *= -1
                    content.append(str(tracked_data.flatten().astype(int).tolist()))
                    
                content = ";".join(content)
                sock.sendto(str.encode(str(content)), serverAddressPort)
                    

            for cam_idx in range(0, len(data.cameras)) :
                camera = input_frame.views[cam_idx].camera
                
                # plot projected get
                projected_gt_dict = {}
                for hand_idx in gt_tracking.keys() :
                    keypoints_world = landmarks_from_hand_pose(
                        hand_model, gt_tracking[hand_idx], hand_idx
                    )
                    keypoints_window = camera.eye_to_window(
                        camera.world_to_eye(keypoints_world)
                    )
                    projected_gt_dict[hand_idx] = keypoints_window
                    
                    for point in keypoints_window :
                        cv2.circle(
                            stereo_bgr_frame_array[frame_idx, cam_idx], 
                            tuple(point.astype(int)), 
                            2,
                            gt_handedness_color_map[hand_idx], 
                            -1
                        )
                    for con in HAND_CONNECTION_MAP :
                        cv2.line(
                            stereo_bgr_frame_array[frame_idx, cam_idx], 
                            tuple(keypoints_window[con[0]].astype(int)), 
                            tuple(keypoints_window[con[1]].astype(int)), 
                            gt_handedness_color_map[hand_idx], 
                            2
                        )
                    
                
                
                # plot reporojected
                projected_keypoints_dict = {}
                for hand_idx in res.hand_poses.keys() :
                    keypoints_world = landmarks_from_hand_pose(
                        hand_model, res.hand_poses[hand_idx], hand_idx
                    )
                    keypoints_window = camera.eye_to_window(
                        camera.world_to_eye(keypoints_world)
                    )
                    projected_keypoints_dict[hand_idx] = keypoints_window
                    
                    for point in keypoints_window :
                        cv2.circle(
                            stereo_bgr_frame_array[frame_idx, cam_idx], 
                            tuple(point.astype(int)), 
                            2,
                            ume_handedness_color_map[hand_idx], 
                            -1
                        )
                        
                    for con in HAND_CONNECTION_MAP :
                        cv2.line(
                            stereo_bgr_frame_array[frame_idx, cam_idx], 
                            tuple(keypoints_window[con[0]].astype(int)), 
                            tuple(keypoints_window[con[1]].astype(int)), 
                            ume_handedness_color_map[hand_idx], 
                            2
                        )
                        
                    cv2.imshow(f"cam_{cam_idx}", stereo_bgr_frame_array[frame_idx, cam_idx])
            key = cv2.waitKey(1)
            if key == ord('q') :
                break




if __name__ == '__main__':
    root = os.path.dirname(__file__)
    model_name = "pretrained_weights.torch"
    model_path = os.path.join(root, "pretrained_models", model_name)

    error_tensors = []
    input_dir = os.path.join(root, "UmeTrack_data", "raw_data", "real")
    output_dir = os.path.join(root, "tmp", "eval_results_known_skeleton", "real")
    pool_size = 8
    
    UMETRACK_ROOT = "."
    VIDEO_PATH = os.path.join(UMETRACK_ROOT, "sample_data/user05/recording_02.mp4")
    DATA_PATH = os.path.join(UMETRACK_ROOT, "sample_data/user05/recording_02.json")
    
    _track_sequence(VIDEO_PATH, DATA_PATH, model_path, override=True)
    