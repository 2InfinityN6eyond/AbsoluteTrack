import time
import os
import json
import time
import cv2
import mediapipe as mp
import numpy as np
import socket

import torch
from typing import Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass
from typing import List

import sys
sys.path.append('..')

import lib.data_utils.fs as fs
from lib.tracker.tracking_result import SingleHandPose
from lib.tracker.perspective_crop import landmarks_from_hand_pose
from lib.tracker.tracker import HandTracker, HandTrackerOpts, ViewData, InputFrame
from lib.models.model_loader import load_pretrained_model
from lib.common.camera import CameraModel, read_camera_from_json




HAND_CONNECTION_MAP = [
    [5, 6], [6, 7], [7, 0], # thumb
    [5, 8], [8, 9], [9, 10], [10, 1], # index
    [5, 11], [11, 12], [12, 13], [13, 2], # middle
    [5, 14], [14, 15], [15, 16], [16, 3], # ring
    [5, 17], [17, 18], [18, 19], [19, 4], # pinky
    [8, 11], [11, 14], [14, 17]
]

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

if __name__ == "__main__":
    UMETRACK_ROOT = ".."
    

    VID_NAME = "recording_00"
    SAMPLE_VID_PATH = os.path.join(UMETRACK_ROOT, "sample_data/user05/", VID_NAME + ".mp4")
    SAMPLE_LABEL_PATH = os.path.join(UMETRACK_ROOT, "sample_data/user05/", VID_NAME + ".json")
    
    model_name = "pretrained_weights.torch"
    model_path = os.path.join(UMETRACK_ROOT, "pretrained_models", model_name)


    model = load_pretrained_model(model_path)
    model.eval()
    tracker_opts = HandTrackerOpts()
    # tracker_opts.hand_ratio_in_crop = 0.5 
    tracker = HandTracker(model, tracker_opts)

    # draw hand pose
    gt_handedness_color_map = {
        0: (0, 0, 255), # RED
        1: (0, 255, 0), # GREEN
    }
    ume_handedness_color_map = {
        0: (10, 10, 55), # RED
        1: (10, 55, 10), # GREEN
    }

    while True :

        cap = cv2.VideoCapture(SAMPLE_VID_PATH)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        data = _load_hand_pose_labels(SAMPLE_LABEL_PATH)
        hand_model = data.hand_model

        fps_outer = 0
        fps_inner = 0
        for frame_idx in range(n_frames):
            stt = time.time()
            
            ret, frame_stereo = cap.read()
            if not ret:
                break

            cam_left = data.cameras[1].copy(
                camera_to_world_xf=data.camera_to_world_transforms[frame_idx, 1]
            )
            cam_right = data.cameras[2].copy(
                camera_to_world_xf=data.camera_to_world_transforms[frame_idx, 2]
            )

            gt_tracking = {}
            for hand_idx in range(0, 2):
                if data.hand_confidences[frame_idx, hand_idx] > 0:
                    gt_tracking[hand_idx] = SingleHandPose(
                        joint_angles=data.joint_angles[frame_idx, hand_idx],
                        wrist_xform=data.wrist_transforms[frame_idx, hand_idx],
                        hand_confidence=data.hand_confidences[frame_idx, hand_idx]
                    )

            # multi_view_images = frame_stereo.reshape(
            #     frame_stereo.shape[0], len(data.cameras), -1
            # )

            multi_view_images_bgr = np.array([
                frame_stereo[:, :frame_width, :],
                frame_stereo[:, frame_width:2*frame_width, :],
                frame_stereo[:, 2*frame_width:3*frame_width, :],
                frame_stereo[:, 3*frame_width:, :],
            ])

            multi_view_images_mono = np.array([
                frame_stereo[:, :frame_width, 0],
                frame_stereo[:, frame_width:2*frame_width, 0],
                frame_stereo[:, 2*frame_width:3*frame_width, 0],
                frame_stereo[:, 3*frame_width:, 0],
            ])

            invalid_camera_to_world = (
                data.camera_to_world_transforms[frame_idx].sum() == 0
            )
            if invalid_camera_to_world:
                assert (
                    not gt_tracking
                ), f"Cameras are not tracked, expecting no ground truth tracking!"

            views = []
            for cam_idx in range(0, len(data.cameras)):
                cur_camera = data.cameras[cam_idx].copy(
                    camera_to_world_xf=data.camera_to_world_transforms[frame_idx, cam_idx]
                )

                views.append(
                    ViewData(
                        image=multi_view_images_mono[cam_idx],
                        camera=cur_camera,
                        camera_angle=data.camera_angles[cam_idx],
                    )
                )

            input_frame = InputFrame(views=views)


            crop_camera_dict = tracker.gen_crop_cameras(
                [view.camera for view in input_frame.views],
                data.camera_angles,
                hand_model,
                gt_tracking,
                min_num_crops=1,
            )

            res = tracker.track_frame_analysis(
                input_frame, 
                hand_model, 
                crop_camera_dict,
                None
            )

            # res = tracker.track_frame(    
            #     input_frame, 
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
                    tracked_data = tracked_keypoints_dict[hand_idx].copy()
                    tracked_data[:, :2] *= -1
                    content.append(str(tracked_data.flatten().astype(int).tolist()))
                
                content = ";".join(content)
                
                sock.sendto(str.encode(str(content)), serverAddressPort)
            
            projected_keypoints_dict = {}
            for cam_idx in range(len(input_frame.views)):
                camera = input_frame.views[cam_idx].camera
                per_cam_projected_keypoints_dict = {}
                for hand_idx in tracked_keypoints_dict.keys():
                    tracked_keypoints = tracked_keypoints_dict[hand_idx]
                    projected_keypoints = camera.eye_to_window(
                        camera.world_to_eye(tracked_keypoints)
                    )
                    per_cam_projected_keypoints_dict[hand_idx] = projected_keypoints
                projected_keypoints_dict[cam_idx] = per_cam_projected_keypoints_dict

            
            fps_inner = 0.5 * fps_inner + 0.5 * (1 / (time.time() - stt))

            for cam_idx, view in enumerate(input_frame.views) :
                camera = view.camera
                image = multi_view_images_bgr[cam_idx]
                cv2.putText(image, str(frame_idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, str(fps_inner), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # plot gt hand pose
                for hand_idx in gt_tracking.keys() :
                    keypoints_world = landmarks_from_hand_pose(
                        hand_model, gt_tracking[hand_idx], hand_idx
                    )
                    keypoints_window = camera.eye_to_window(camera.world_to_eye(keypoints_world))
                    
                    for point in keypoints_window :
                        x, y = point
                        cv2.circle(image, (int(x), int(y)), 2, gt_handedness_color_map[hand_idx], -1)

                    for con in HAND_CONNECTION_MAP :
                        cv2.line(
                            image, 
                            keypoints_window[con[0]].astype(np.int32), 
                            keypoints_window[con[1]].astype(np.int32), 
                            gt_handedness_color_map[hand_idx],
                            1
                        )

                # plot umetrack hand pose
                for hand_idx in projected_keypoints_dict[cam_idx].keys() :
                    keypoints_window = projected_keypoints_dict[cam_idx][hand_idx]
                    for point in keypoints_window :
                        x, y = point
                        cv2.circle(image, (int(x), int(y)), 2, ume_handedness_color_map[hand_idx], -1)

                    for con in HAND_CONNECTION_MAP :
                        cv2.line(
                            image,
                            keypoints_window[con[0]].astype(np.int32),
                            keypoints_window[con[1]].astype(np.int32),
                            ume_handedness_color_map[hand_idx],
                            2
                        )

                cv2.imshow(f"cam{cam_idx}", image)
            k = cv2.waitKey(1)
            if k==49:
                break

            fps_outer = 0.5 * fps_outer + 0.5 * (1 / (time.time() - stt))
            # print(f"FPS: {fps_outer:.2f}, {fps_inner:.2f}")
            
        
    cap.release()
    cv2.destroyAllWindows()