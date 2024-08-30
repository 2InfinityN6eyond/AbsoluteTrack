import multiprocessing as mp
import numpy as np
import cv2
import time

import mediapipe

class MediaPipeEstimator(mp.Process):
    def __init__(
        self,
        config,
        camera_idx,
        shared_array_rgb,
        shared_mp_pose_array,
        camreader2mp,
        mp2ume,
        stop_event
    ):
        super().__init__()
        self.config = config
        self.camera_idx = camera_idx
        self.shared_array_rgb = shared_array_rgb
        self.shared_mp_pose_array = shared_mp_pose_array
        self.camreader2mp = camreader2mp
        self.mp2ume = mp2ume
        self.stop_event = stop_event

    def run(self):
        # initialize image shared array
        self._shared_array_rgb_list = [
            np.frombuffer(
                self.shared_array_rgb[i].buf, dtype=np.uint8
            ).reshape((
                2 if self.config.is_stereo else 1,
                self.config.camera.image_height,
                self.config.camera.image_width,
                3
            )) for i in range(self.config.buffer.size)
        ]
        self.shared_array_rgb = list(map(
            lambda x : x[self.camera_idx],
            self._shared_array_rgb_list
        ))
        
        # initialize hand shared array
        self._shared_mp_pose_list = [
            np.frombuffer(
                self.shared_mp_pose_array[i].buf, dtype=np.float32
            ).reshape((
                2 if self.config.is_stereo else 1,
                self.config.media_pipe.max_num_hands,
                self.config.media_pipe.num_keypoints,
                3
            )) for i in range(self.config.buffer.size)
        ]
        self.shared_mp_pose = list(map(
            lambda x : x[self.camera_idx],
            self._shared_mp_pose_list
        ))
        
        mp_hands = mediapipe.solutions.hands
        # initialize hand detector list for each camera
        mp_hand_detector = mp_hands.Hands(
            max_num_hands       = self.config.media_pipe.max_num_hands,
            model_complexity    = self.config.media_pipe.model_complexity,
            min_detection_confidence= self.config.media_pipe.min_detection_confidence,
            min_tracking_confidence = self.config.media_pipe.min_tracking_confidence
        )
        
        fps1 = 0
        fps2 = 0
        while not self.stop_event.is_set():
            stt1 = time.time()
            
            index = self.camreader2mp.get()
            
            stt2 = time.time()
            
            frame = self.shared_array_rgb[index]
            
            # set keypoint values to zero
            self.shared_mp_pose[index][:] = 0
            
            results = mp_hand_detector.process(frame)
            
            mp_hand_pose_dict = {}
            
            if results.multi_handedness:
                for handedness, landmark in zip(
                    results.multi_handedness,
                    results.multi_hand_landmarks
                ) :
                    # Singe Egocentric
                    # In UmeTrack, RIGHT_HAND_INDEX = 1
                    # if right hand, UmeTrack flips it before processing
                    # 
                    hand_index = 1 - handedness.classification[0].index
                    hand_pose = np.array(list(map(
                        lambda l : [l.x, l.y, l.z],
                        landmark.landmark
                    ))) * np.array([
                        self.config.camera.image_width,
                        self.config.camera.image_height,
                        self.config.camera.image_width
                    ])
                    #print("recording", self.camera_idx, hand_index)
                    mp_hand_pose_dict[hand_index] = hand_pose
                    
                    self.shared_mp_pose[hand_index][:, :] = hand_pose

                    # for point in hand_pose:
                    #     x, y, z = point
                    #     cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        
            # cv2.putText(frame, f"FPS: {fps1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imshow(f"Camera {self.camera_idx}", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            # key = cv2.waitKey(1)
            
            fps1 = 0.5 * fps1 + 0.5 * (1 / (time.time() - stt1))
            fps2 = 0.5 * fps2 + 0.5 * (1 / (time.time() - stt2))
            # print(f"camera {self.camera_idx} FPS: {int(fps1)}, {int(fps2)}")
        
            self.mp2ume.put((index, mp_hand_pose_dict))


"""
class MediaPipeEstimator(mp.Process):
    def __init__(
        self,
        config, 
        shared_array_rgb,
        image_ready_condition,
        shared_image_index,
        shared_mp_pose_array,
        mp_pose_ready_condition,
        shared_mp_pose_index,
        stop_event
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb = shared_array_rgb
        self.image_ready_condition = image_ready_condition
        self.shared_image_index = shared_image_index
        self.shared_mp_pose_array = shared_mp_pose_array
        self.mp_pose_ready_condition = mp_pose_ready_condition
        self.shared_mp_pose_index = shared_mp_pose_index
        self.stop_event = stop_event

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
            max_num_hands       = self.config.hand_pose_estimator.max_num_hands,
            model_complexity    = self.config.hand_pose_estimator.model_complexity,
            min_detection_confidence= self.config.hand_pose_estimator.min_detection_confidence,
            min_tracking_confidence = self.config.hand_pose_estimator.min_tracking_confidence
        ) for _ in range(2 if self.config.is_stereo else 1)]        
        
        # initialize hand shared array
        self.shared_mp_pose_list = [
            np.frombuffer(
                self.shared_mp_pose_array[i].buf, dtype=np.float32
            ).reshape((
                2 if self.config.is_stereo else 1,
                self.config.hand_pose_estimator.max_num_hands,
                self.config.hand_pose_estimator.num_keypoints,
                3
            )) for i in range(self.config.buffer.size)
        ]
        
        fps1 = 0
        fps2 = 0
        while not self.stop_event.is_set():
            stt1 = time.time()
            with self.image_ready_condition:
                self.image_ready_condition.wait()
                
                stt2 = time.time()
                
                index = self.shared_image_index.value
                frame = self.shared_array_rgb_list[index]
                
                # set keypoint values to zero
                self.shared_mp_pose_list[index][:] = 0
                
                for cam_idx, (frame, detector) in enumerate(zip(frame, mp_hand_detector_list)):
                    
                    results = detector.process(frame)
                    
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
                        
                            self.shared_mp_pose_list[index][cam_idx, hand_index, :, :] = hand_pose
                
                fps1 = 0.5 * fps1 + 0.5 * (1 / (time.time() - stt1))
                fps2 = 0.5 * fps2 + 0.5 * (1 / (time.time() - stt2))
                print(f"FPS: {int(fps1)}, {int(fps2)}")
                        
                with self.mp_pose_ready_condition:
                    self.shared_mp_pose_index.value = index
                    self.mp_pose_ready_condition.notify_all()

"""