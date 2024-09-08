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
    min_detection_confidence: float = 0.4
    min_tracking_confidence: float = 0.4
    
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

        print("shared_array_rgb_names image visualizer", self.shared_array_rgb_names)
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
        mp_handedness_color_map = {
            0: (255, 0, 0), # red
            1: (0, 255, 0), # green
        }
        
        ume_handedness_color_map = {
            0: (150, 100, 100), # red
            1: (100, 150, 100), # green
        }
        
        
        fps = 0
        while not self.stop_event.is_set():
            if self.ume2imgviz.empty():
                continue
            
            stt = time.time()        
            (
                index, 
                mp_hand_pose_dict, 
                tracked_keypoints_dict,
                projected_keypoints_dict
            ) = self.ume2imgviz.get()

            frame = self.shared_array_rgb_list[index].copy()
            
            # cv2.imshow("frame", frame[0])
            
            mp_pose_dict = mp_hand_pose_dict
            
            for cam_idx in range(2 if self.config.is_stereo else 1):
                img = frame[cam_idx]
            
                for hand_idx, hand_pose in mp_pose_dict[cam_idx].items():
                    if hand_pose.mean() > 0.1:
                        for keypoint in hand_pose:
                            x, y, z = keypoint
                            cv2.circle(
                                img, (int(x), int(y)), 5,
                                mp_handedness_color_map[hand_idx], -1
                            )
                            
                for hand_idx, hand_pose in projected_keypoints_dict[cam_idx].items():
                    if hand_pose.mean() > 0.1:
                        for keypoint in hand_pose:
                            x, y = keypoint
                            cv2.circle(
                                img, (int(x), int(y)), 5, 
                                ume_handedness_color_map[hand_idx], -1
                            )
                
                cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(f"Camera {cam_idx}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            
            fps = 0.5 * fps + 0.5 * (1 / (time.time() - stt))
            
            if self.verbose:
                print(f"                                         VIS IDX {index} FPS: {int(fps)} IDX2 {self.shared_array_rgb_list[index][0, 0, 0, 2]}")
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
    shared_array_rgb_names = [shm.name for shm in shared_array_rgb]

    print("shared_array_rgb_names", shared_array_rgb_names)

    
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
    shared_array_mono_names = [shm.name for shm in shared_array_mono]
    
    print("shared_array_mono_names", shared_array_mono_names)
    stop_event = mp.Event()
    
    mp2ume = mp.Queue()
    ume2imgviz = mp.Queue()
    
    media_pipe_estimator = MediaPipeEstimator(
        config                  = config,
        shared_array_rgb_names = shared_array_rgb_names,
        shared_array_mono_names = shared_array_mono_names,
        mp2ume                  = mp2ume,
        stop_event              = stop_event,
        verbose                 = True
    )
    ume_tracker = UmeTracker(
        config                  = config, 
        shared_array_mono_names = shared_array_mono_names,
        mp2ume                  = mp2ume,
        ume2imgviz              = ume2imgviz,
        stop_event              = stop_event,
        verbose                 = True
    )
    
    img_visualizer = ImageVisualizer(
        config                  = config,
        shared_array_rgb_names = shared_array_rgb_names,
        ume2imgviz              = ume2imgviz,
        stop_event              = stop_event,
        verbose                 = True
    )

    processes = [
        img_visualizer,
        ume_tracker,
        media_pipe_estimator,
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