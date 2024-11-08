import time
import numpy as np
import cv2
import multiprocessing as mp
from copy import deepcopy
from multiprocessing import shared_memory

import mediapipe

from const_values import (
    MP_CONNECTION_MAP, 
    HAND_CONNECTION_MAP, 
    MP_HANDEDNESS_COLOR_MAP, 
    UME_HANDEDNESS_COLOR_MAP
)


class MediaPipeWorker(mp.Process):
    def __init__(
        self,
        view_idx,
        shm_name_list,
        config,
        in_queue,
        out_queue,
        stop_event,
        verbose = False 
    ):
        super().__init__()
        self.view_idx = view_idx
        self.shm_name_list = shm_name_list
        self.config = config
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.verbose = verbose
        
    def run(self):
        self.shm_list = [
            shared_memory.SharedMemory(
                name = shm_name
            ) for shm_name in self.shm_name_list
        ]
        self.rgb_frames_array_list = [
            np.ndarray(
                shape = (
                    self.config.camera.n_views,
                    self.config.camera.image_height,
                    self.config.camera.image_width,
                    3
                ),
                dtype = np.uint8,
                buffer = shm.buf
            ) for shm in self.shm_list
        ]

        mp_hands = mediapipe.solutions.hands
        mp_hand_detector = mp_hands.Hands(
            max_num_hands       = self.config.media_pipe.max_num_hands,
            model_complexity    = self.config.media_pipe.model_complexity,
            min_detection_confidence= self.config.media_pipe.min_detection_confidence,
            min_tracking_confidence = self.config.media_pipe.min_tracking_confidence
        )

        while not self.stop_event.is_set():
            if not self.in_queue.empty():
                index = self.in_queue.get()
                
                frame = self.rgb_frames_array_list[index][self.view_idx].copy()
            
                results = mp_hand_detector.process(frame)


                mp_pose_dict = {}
                if results.multi_handedness:
                    # for hand_idx, hand_handedness in enumerate(results.multi_handedness):
                    #     handedness_class = hand_handedness.classification[0].label
                    #     handedness_score = hand_handedness.classification[0].score
                    #     print(f"Hand {hand_idx} is {handedness_class} with confidence {handedness_score}")

                    for handedness, landmark in zip(
                        results.multi_handedness,
                        results.multi_hand_landmarks
                    ) :
                        hand_index =  1 - handedness.classification[0].index
                        confidence = handedness.classification[0].score
                        hand_pose = np.array(list(map(
                            lambda l : [l.x, l.y, l.z],
                            landmark.landmark
                        ))) * np.array([
                            self.config.camera.image_width,
                            self.config.camera.image_height,
                            self.config.camera.image_width
                        ])
                        mp_pose_dict[hand_index] = hand_pose


                        # for pos in mp_pose_dict[hand_index]:
                        #     cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, MP_HANDEDNESS_COLOR_MAP[hand_index], -1)

                        # for connection in MP_CONNECTION_MAP:
                        #     cv2.line(frame, (int(mp_pose_dict[hand_index][connection[0]][0]), int(mp_pose_dict[hand_index][connection[0]][1])), (int(mp_pose_dict[hand_index][connection[1]][0]), int(mp_pose_dict[hand_index][connection[1]][1])), MP_HANDEDNESS_COLOR_MAP[hand_index], 2)

                self.out_queue.put((index, mp_pose_dict))

                # cv2.imshow(f"view {self.view_idx}", frame)
                # cv2.waitKey(1)
                


class MediaPipeEstimator(mp.Process):
    def __init__(
        self,
        config,
        rgb_frames_shm_name_list,
        cam2mp,
        mp2ume,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.rgb_frames_shm_name_list = rgb_frames_shm_name_list
        self.cam2mp = cam2mp
        self.mp2ume = mp2ume
        self.stop_event = stop_event
        self.verbose = verbose
        
    def run(self):

        in_queue_list = [
            mp.Queue() for _ in range(self.config.camera.n_views)
        ]
        out_queue_list = [
            mp.Queue() for _ in range(self.config.camera.n_views)
        ]

        worker_list = [
            MediaPipeWorker(
                view_idx = view_idx,
                shm_name_list = self.rgb_frames_shm_name_list,
                config = self.config,
                in_queue = in_queue_list[view_idx],
                out_queue = out_queue_list[view_idx],
                stop_event = self.stop_event,
                verbose = self.verbose
            ) for view_idx in range(self.config.camera.n_views)
        ]
        for p in worker_list:
            p.start()


        while not self.stop_event.is_set():
            if not self.cam2mp.empty():
                index = self.cam2mp.get()
                
                result_dict = {}
                for view_idx in range(self.config.camera.n_views):
                    in_queue_list[view_idx].put(index)
                for view_idx in range(self.config.camera.n_views):
                    (buffer_idx, mp_pose_dict) = out_queue_list[view_idx].get()

                    result_dict[view_idx] = mp_pose_dict
                    
                    if self.verbose:
                        print(result_dict)
                        print()

                # self.mp2ume.put(index, result_dict)
                self.mp2ume.put((index, deepcopy(result_dict)))
