import numpy as np
import cv2
import time

import multiprocessing as mp
from multiprocessing import shared_memory

from const_values import (
    MP_CONNECTION_MAP, 
    HAND_CONNECTION_MAP, 
    MP_HANDEDNESS_COLOR_MAP, 
    UME_HANDEDNESS_COLOR_MAP
)

class ImageVisualizer(mp.Process):
    def __init__(
        self,
        config,
        shared_array_rgb_names,
        ume2imgviz,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb_names = shared_array_rgb_names
        self.ume2imgviz = ume2imgviz
        self.stop_event = stop_event
        self.verbose = verbose
        
    def run(self):
        
        self.rgb_shm_list = [
            shared_memory.SharedMemory(
                name = name
            ) for name in self.shared_array_rgb_names
        ]
        self.shared_array_rgb_list = [
            np.ndarray(
                (
                    self.config.camera.n_views,
                    self.config.camera.image_height, 
                    self.config.camera.image_width,
                    3
                ),
                dtype=np.uint8,
                buffer=shm.buf
            ) for shm in self.rgb_shm_list
        ]
        

        fps = 0
        while not self.stop_event.is_set():
            
            stt = time.time()
            
            (
                index, 
                mp_hand_pose_dict, 
                tracked_keypoints_dict,
                projected_keypoints_dict
            ) = self.ume2imgviz.get()

            multiview_frame = self.shared_array_rgb_list[index].copy()
            
            for cam_idx in range(self.config.camera.n_views):
                frame = multiview_frame[cam_idx]
                
                for hand_idx, hand_pose in mp_hand_pose_dict[cam_idx].items():
                    if hand_pose.mean() > 0.1:
                        for keypoint in hand_pose:
                            x, y, z = keypoint
                            cv2.circle(
                                frame, (int(x), int(y)), 2,
                                MP_HANDEDNESS_COLOR_MAP[hand_idx], -1
                            )
                        
                        for con in MP_CONNECTION_MAP:
                            cv2.line(
                                frame, (int(hand_pose[con[0]][0]), int(hand_pose[con[0]][1])), 
                                (int(hand_pose[con[1]][0]), int(hand_pose[con[1]][1])), 
                                MP_HANDEDNESS_COLOR_MAP[hand_idx], 1
                            )
                
                for hand_idx, hand_pose in projected_keypoints_dict[cam_idx].items():
                    if hand_pose.mean() > 0.1:
                        for keypoint in hand_pose:
                            x, y = keypoint
                            cv2.circle(
                                frame, (int(x), int(y)), 2, 
                                UME_HANDEDNESS_COLOR_MAP[hand_idx], -1
                            )
                        
                        for con in HAND_CONNECTION_MAP:
                            cv2.line(
                                frame, (int(hand_pose[con[0]][0]), int(hand_pose[con[0]][1])), 
                                (int(hand_pose[con[1]][0]), int(hand_pose[con[1]][1])), 
                                UME_HANDEDNESS_COLOR_MAP[hand_idx], 1
                            )
                
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(f"Camera {cam_idx}", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            
            fps = 0.5 * fps + 0.5 * (1 / (time.time() - stt))
            
            if self.verbose:
                print(f"                                         VIS IDX {index} FPS: {int(fps)} IDX2 {self.shared_array_rgb_list[index][0, 0, 0, 2]}")
        cv2.destroyAllWindows()           
