import numpy as np
import cv2
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import mediapipe

class MediaPipeEstimator(mp.Process):
    def __init__(
        self,
        config,
        shared_array_rgb_names,
        shared_array_mono_names,
        mp2ume,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb_names = shared_array_rgb_names
        self.shared_array_mono_names = shared_array_mono_names
        self.mp2ume = mp2ume
        self.stop_event = stop_event
        self.verbose = verbose
        
    def run(self):
        
        # open camera
        cap = cv2.VideoCapture(self.config.camera.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.image_width * (2 if self.config.is_stereo else 1))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.image_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
                
        # initialize image shared array
        # self.shared_array_rgb_list = [
        #     np.frombuffer(
        #         self.shared_array_rgb[i].buf, dtype=np.uint8
        #     ).reshape((
        #         2 if self.config.is_stereo else 1,
        #         self.config.camera.image_height,
        #         self.config.camera.image_width,
        #         3
        #     )) for i in range(self.config.buffer.size)
        # ]
        # self.shared_array_mono_list = [
        #     np.frombuffer(
        #         self.shared_array_mono[i].buf, dtype=np.uint8
        #     ).reshape((
        #         2 if self.config.is_stereo else 1,
        #         self.config.camera.image_height,
        #         self.config.camera.image_width,
        #     )) for i in range(self.config.buffer.size)
        # ]

        self.rgb_shm_list = [
            shared_memory.SharedMemory(
                name = name
            ) for name in self.shared_array_rgb_names
        ]
        self.shared_array_rgb_list = [
            np.ndarray(
                (
                    2 if self.config.is_stereo else 1,
                    self.config.camera.image_height,
                    self.config.camera.image_width,
                    3
                ),
                dtype=np.uint8,
                buffer=shm.buf
            ) for shm in self.rgb_shm_list
        ]

        self.mono_shm_list = [
            shared_memory.SharedMemory(
                name = name
            ) for name in self.shared_array_mono_names
        ]
        self.shared_array_mono_list = [
            np.ndarray(
                (
                    2 if self.config.is_stereo else 1, 
                    self.config.camera.image_height, 
                    self.config.camera.image_width
                ),
                dtype=np.uint8,
                buffer=shm.buf
            ) for shm in self.mono_shm_list
        ]
        
        mp_hands = mediapipe.solutions.hands
        mp_hand_detector_list = [mp_hands.Hands(
            max_num_hands       = self.config.media_pipe.max_num_hands,
            model_complexity    = self.config.media_pipe.model_complexity,
            min_detection_confidence= self.config.media_pipe.min_detection_confidence,
            min_tracking_confidence = self.config.media_pipe.min_tracking_confidence
        ) for _ in range(2 if self.config.is_stereo else 1)]        
    
        
        fps = 0
        index = 0
        
        idx2 = 0
        
        while not self.stop_event.is_set():
            stt1 = time.time()
            
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.config.camera.flip_horizontal:
                frame = cv2.flip(frame, 1)
            if self.config.camera.flip_vertical:
                frame = cv2.flip(frame, 0)
            
            frame_mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # handle stereo vs single view
            if self.config.is_stereo:
                frame_rgb = np.array([
                    frame[:, :self.config.camera.image_width], 
                    frame[:, self.config.camera.image_width:]
                ])
                frame_mono = np.array([
                    frame_mono[:, :self.config.camera.image_width],
                    frame_mono[:, self.config.camera.image_width:] 
                ])
            else:
                frame_rgb = np.array([frame])
                frame_mono = np.array([frame_mono])

            self.shared_array_rgb_list[index][:] = frame_rgb.copy()
            self.shared_array_mono_list[index][:] = frame_mono.copy()
            
            
            index += 1
            index %= self.config.buffer.size
            
    
            mp_pose_dict = {}
            for cam_idx, (frame, detector) in enumerate(zip(frame_rgb, mp_hand_detector_list)):
                
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
                    
                mp_pose_dict[cam_idx] = mp_pose_dict_per_cam   
            
            fps = 0.5 * fps + 0.5 * (1 / (time.time() - stt1))
            if self.verbose:
                
                if self.config.is_stereo:
                    self.shared_array_rgb_list[index][0, 0, 0, :] = idx2
                    self.shared_array_mono_list[index][0, 0, 0] = idx2
                    
                    cv2.imshow("frame mp", frame_rgb[0])
                    cv2.waitKey(1)
                
                    print(f"MP FPS:{int(fps)} IDX:{index} IDX2:{self.shared_array_rgb_list[index][0, 0, 0, 0]}")
            
                    idx2 += 1
            
            self.mp2ume.put((
                index,
                mp_pose_dict
            ))