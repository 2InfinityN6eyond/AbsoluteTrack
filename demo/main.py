import multiprocessing as mp
import numpy as np
import cv2
import tyro
import time

from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing import Event, Condition
from multiprocessing import Manager

import mediapipe

from ume_tracker import UmeTracker
from mediapipe_estimator import MediaPipeEstimator
@dataclass
class CameraConfig:
    image_height: int = 480
    image_width: int = 640
    fps: int = 30  # Frames per second for camera
    camera_index: int = 0  # Default camera index
    flip_horizontal: bool = False
    flip_vertical: bool = False

@dataclass
class BufferConfig:
    size: int = 6

@dataclass
class MediaPipeConfig :
    max_num_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    num_keypoints: int = 21
    
@dataclass
class UmeTrackerConfig:
    max_num_hands: int = 2
    num_keypoints: int = 21
    
@dataclass
class Config:
    is_stereo: bool = True
    camera: CameraConfig = CameraConfig()
    buffer: BufferConfig = BufferConfig()
    media_pipe: MediaPipeConfig = MediaPipeConfig()
    ume_tracker: UmeTrackerConfig = UmeTrackerConfig()


class CameraReader(mp.Process):
    def __init__(
        self,
        config,
        shared_array_rgb,
        shared_array_mono,
        camreader2mp_list,
        stop_event
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb = shared_array_rgb
        self.shared_array_mono = shared_array_mono
        self.camreader2mp_list = camreader2mp_list
        self.stop_event = stop_event

    def run(self):
        cap = cv2.VideoCapture(self.config.camera.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.image_width * (2 if self.config.is_stereo else 1))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.image_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        # initialize list of shared array
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
        self.shared_array_mono_list = [
            np.frombuffer(
                self.shared_array_mono[i].buf, dtype=np.uint8
            ).reshape((
                2 if self.config.is_stereo else 1, 
                self.config.camera.image_height, 
                self.config.camera.image_width,
            )) for i in range(self.config.buffer.size)
        ]

        index = 0
        while not self.stop_event.is_set():
            
            frame_index = index % len(self.shared_array_rgb)
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

            self.shared_array_rgb_list[frame_index][:] = frame_rgb.copy()
            self.shared_array_mono_list[frame_index][:] = frame_mono.copy()
    
            for queue in self.camreader2mp_list:
                queue.put(index)
                
            index += 1
            index %= len(self.shared_array_rgb)
            
        cap.release()


class ImageVisualizer(mp.Process):
    def __init__(
        self,
        config,
        shared_array_rgb,
        shared_mp_pose_array,
        ume2imgviz,
        stop_event
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb = shared_array_rgb
        self.shared_mp_pose_array = shared_mp_pose_array
        self.ume2imgviz = ume2imgviz
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
        
            (
                index, 
                mp_hand_pose_dict1, mp_hand_pose_dict2, 
                tracked_keypoints_dict,
                projected_keypoints_dict
            ) = self.ume2imgviz.get()
            
            stt2 = time.time()
            
            frame = self.shared_array_rgb_list[index]
            mp_pose_results = self.shared_mp_pose_list[index]
            
            mp_pose_dict = {
                0: mp_hand_pose_dict1,
                1: mp_hand_pose_dict2,
            }
            
            for cam_idx in range(2 if self.config.is_stereo else 1):
                img = frame[cam_idx]
                for hand_idx, hand_pose in mp_pose_dict[cam_idx].items():
                    if hand_pose.mean() > 0.1:
                        
                        for keypoint in hand_pose:
                            x, y, z = keypoint
                            cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
                
                
                for hand_idx, hand_pose in projected_keypoints_dict[cam_idx].items():
                    if hand_pose.mean() > 0.1:
                        for keypoint in hand_pose:
                            x, y, z = keypoint
                            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                
                
                # mp_pose_results_cam = mp_pose_results[cam_idx]
                                
                # for hand_idx, hand_keypoint in enumerate(mp_pose_results_cam):
                #     if hand_keypoint.mean() > 0.1:
                #         print("recorded", cam_idx, hand_idx)
                #         for keypoint in hand_keypoint:
                #             x, y, z = keypoint
                #         cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

                cv2.putText(img, f"FPS: {fps1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(f"Camera {cam_idx}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
            fps2 = 0.5 * fps2 + 0.5 * (1 / (time.time() - stt2))
            fps1 = 0.5 * fps1 + 0.5 * (1 / (time.time() - stt1))
            # print(f"FPS: {int(fps1)}, {int(fps2)}")
        cv2.destroyAllWindows()           

if __name__ == "__main__":
    # Use tyro to parse command-line arguments into Config
    config = tyro.cli(Config)

    # Create shared memory for the RGB image 
    shared_array_rgb = [
        shared_memory.SharedMemory(
            create=True, size=np.prod((
                2 if config.is_stereo else 1,
                config.camera.image_height,
                config.camera.image_width,
                3
            ))
        ) for _ in range(config.buffer.size)
    ]
    
    # Create shared memory for the monochrome images
    shared_array_mono = [
        shared_memory.SharedMemory(
            create=True, size=np.prod((
                2 if config.is_stereo else 1,
                config.camera.image_height,
                config.camera.image_width,
            ))
        ) for _ in range(config.buffer.size)
    ]
    
    # Create shared memory for the mediapipe hand pose results
    shared_mp_pose_array = [
        shared_memory.SharedMemory(
            create=True, size=np.prod((
                2 if config.is_stereo else 1,
                config.media_pipe.max_num_hands,
                config.media_pipe.num_keypoints,
                3,
                4 # float32
            ))
        ) for _ in range(config.buffer.size)
    ]
    
    # Create shared memory for the UmeTrack hand pose results
    shared_ume_pose_array = [
        shared_memory.SharedMemory(
            create=True, size=np.prod((
                config.ume_tracker.max_num_hands,
                config.ume_tracker.num_keypoints,
                3,
                4 # float32
            ))
        ) for _ in range(config.buffer.size)
    ]
    stop_event = mp.Event()
    
    # 코드가 매우 더러움ㅠㅠ
    camreader2mp_list = [mp.Queue() for _ in range(2 if config.is_stereo else 1)]
    mp2ume_list = [mp.Queue() for _ in range(2 if config.is_stereo else 1)]
    ume2imgviz = mp.Queue()
    
    camera_reader = CameraReader(
        config              = config, 
        shared_array_rgb    = shared_array_rgb,
        shared_array_mono   = shared_array_mono,
        camreader2mp_list   = camreader2mp_list,
        stop_event          = stop_event
    )
    media_pipe_estimator_list = [
        MediaPipeEstimator(
            config                  = config,
            camera_idx              = i,
            shared_array_rgb        = shared_array_rgb,
            shared_mp_pose_array    = shared_mp_pose_array,
            camreader2mp            = camreader2mp_list[i],
            mp2ume                  = mp2ume_list[i],
            stop_event              = stop_event
        ) for i in range(2 if config.is_stereo else 1)
    ]
    ume_tracker = UmeTracker(
        config                  = config, 
        shared_array_mono       = shared_array_mono,
        shared_mp_pose_array    = shared_mp_pose_array,
        shared_ume_pose_array   = shared_ume_pose_array,
        mp2ume_list             = mp2ume_list,
        ume2imgviz              = ume2imgviz,
        stop_event              = stop_event
    )
    
    img_visualizer = ImageVisualizer(
        config                  = config,
        shared_array_rgb        = shared_array_rgb,
        shared_mp_pose_array    = shared_mp_pose_array,
        ume2imgviz              = ume2imgviz,
        stop_event              = stop_event
    )

    processes = [
        img_visualizer,
        ume_tracker,
        *media_pipe_estimator_list,
        camera_reader,
    ]
    for p in processes:
        p.start()

    try:
        while True:
            continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_event.set()
        for p in processes:
            p.join()

    # Clean up shared memory
    for shm in shared_array_rgb + shared_array_mono:
        shm.close()
        shm.unlink()