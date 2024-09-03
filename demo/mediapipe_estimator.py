import multiprocessing as mp
import numpy as np
import cv2
import time

import mediapipe


class MediaPipeEstimator(mp.Process):
    def __init__(
        self,
        config,
        shared_array_rgb,
        shared_mp_pose_array,
        camreader2mp,
        mp2ume,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb = shared_array_rgb
        self.shared_mp_pose_array = shared_mp_pose_array
        self.camreader2mp = camreader2mp
        self.mp2ume = mp2ume
        self.stop_event = stop_event
        self.verbose = verbose
    def run(self):
        # initialize image shared array
        self.shared_array_rgb_list = [
            np.frombuffer(
                self.shared_array_rgb[i].buf, dtype=np.uint8
            ).reshape((
                2 if self.config.is_stereo else 1,
                self.config.camera.image_height,
                self.config.camera.image_width,
                3
            )) for i in range(self.config.buffer.size)
        ]
        
        mp_hands = mediapipe.solutions.hands
        # initialize hand detector list for each camera
        mp_hand_detector_list = [mp_hands.Hands(
            max_num_hands       = self.config.media_pipe.max_num_hands,
            model_complexity    = self.config.media_pipe.model_complexity,
            min_detection_confidence= self.config.media_pipe.min_detection_confidence,
            min_tracking_confidence = self.config.media_pipe.min_tracking_confidence
        ) for _ in range(2 if self.config.is_stereo else 1)]        
        
        # initialize hand shared array
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
        
        fps1 = 0
        fps2 = 0
        while not self.stop_event.is_set():
            stt1 = time.time()
            
            if self.camreader2mp.empty():
                continue
            
            index = self.camreader2mp.get()
            
            stt2 = time.time()
            
            frame = self.shared_array_rgb_list[index]
            
            # set keypoint values to zero
            self.shared_mp_pose_list[index][:] = 0
            
            mp_pose_dict = {}
            for cam_idx, (frame, detector) in enumerate(zip(frame, mp_hand_detector_list)):
                
                results = detector.process(frame)
                
                mp_pose_dict_per_cam = {}
                if results.multi_handedness:
                    for handedness, landmark in zip(
                        results.multi_handedness,
                        results.multi_hand_landmarks
                    ) :
                        hand_index =  handedness.classification[0].index
                        hand_pose = np.array(list(map(
                            lambda l : [l.x, l.y, l.z],
                            landmark.landmark
                        ))) * np.array([
                            self.config.camera.image_width,
                            self.config.camera.image_height,
                            self.config.camera.image_width
                        ])
                        
                        mp_pose_dict_per_cam[hand_index] = hand_pose
                    
                        # self.shared_mp_pose_list[index][cam_idx, hand_index, :, :] = hand_pose
            
                mp_pose_dict[cam_idx] = mp_pose_dict_per_cam   
            
            
            fps1 = 0.5 * fps1 + 0.5 * (1 / (time.time() - stt1))
            fps2 = 0.5 * fps2 + 0.5 * (1 / (time.time() - stt2))
            if self.verbose:
                print(f"               MP FPS: {int(fps1)}, {int(fps2)}")
            
            self.mp2ume.put((
                index,
                mp_pose_dict
            ))