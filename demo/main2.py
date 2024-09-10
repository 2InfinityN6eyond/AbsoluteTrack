import time
import tyro
import cv2
import numpy as np
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import freeze_support
from multiprocessing import Process


from camera_reader import CameraReader
from media_pipe_estimator import MediaPipeEstimator
from ume_tracker import UmeTracker
from image_visualizer import ImageVisualizer

import sys


@dataclass
class CameraConfig:
    n_views: int = 2
    fps: int = 30  # Frames per second for camera
    image_height: int = 480
    image_width: int = 640
    cv_camera_index: int = 0  # Default camera index
    flip_horizontal: bool = False
    flip_vertical: bool = False

@dataclass
class BufferConfig:
    size: int = 6

@dataclass
class MediaPipeConfig :
    max_num_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    
    num_keypoints: int = 21
    
@dataclass
class UmeTrackerConfig:
    max_num_hands: int = 2
    num_keypoints: int = 21
    
@dataclass
class Config:
    camera: CameraConfig = CameraConfig()
    buffer: BufferConfig = BufferConfig()
    media_pipe: MediaPipeConfig = MediaPipeConfig()
    ume_tracker: UmeTrackerConfig = UmeTrackerConfig()



if __name__ == "__main__":
    # freeze_support()
    config = tyro.cli(Config)
    
    #print(sys.version)

    try:
        rgb_frames_shm_list = [
            shared_memory.SharedMemory(
                create=True,
                size=int(np.prod((
                    config.camera.n_views, 
                    config.camera.image_height, 
                    config.camera.image_width, 
                    3
                )) * np.dtype(np.uint8).itemsize)
            ) for _ in range(config.buffer.size)
        ]
    except Exception as e:
        print(f"Error creating shared memory: {e}")

    rgb_frames_shm_name_list = [
        shm.name for shm in rgb_frames_shm_list
    ]

    try:
        mono_frames_shm_list = [
            shared_memory.SharedMemory(
                create=True,
                size=int(np.prod((
                    config.camera.n_views,
                    config.camera.image_height,
                    config.camera.image_width
                )) * np.dtype(np.uint8).itemsize)
            ) for _ in range(config.buffer.size)
        ]
    except Exception as e:
        print(f"Error creating shared memory: {e}")

    mono_frames_shm_name_list = [
        shm.name for shm in mono_frames_shm_list
    ]
    

    stop_event = mp.Event()

    cam2mp = mp.Queue()
    mp2ume = mp.Queue()
    ume2vz = mp.Queue()

    
    camera_reader = CameraReader(
        config = config,
        rgb_frames_shm_name_list = rgb_frames_shm_name_list,
        mono_frames_shm_name_list = mono_frames_shm_name_list,
        cam2mp = cam2mp,
        stop_event = stop_event,
        verbose = False
    )
    media_pipe_estimator = MediaPipeEstimator(  
        config = config,
        rgb_frames_shm_name_list = rgb_frames_shm_name_list,
        cam2mp = cam2mp,
        mp2ume = mp2ume,
        stop_event = stop_event,
        verbose = False
    )
    ume_tracker = UmeTracker(
        config = config,
        shm_name_list = mono_frames_shm_name_list,
        mp2ume = mp2ume,
        ume2vz = ume2vz,
        stop_event = stop_event,
        verbose = False
    )
    image_visualizer = ImageVisualizer(
        config = config,
        shared_array_rgb_names = rgb_frames_shm_name_list,
        ume2imgviz = ume2vz,
        stop_event = stop_event,
        verbose = False
    )


    processes = [camera_reader, media_pipe_estimator, ume_tracker ,image_visualizer]
    
    for p in processes:
        print(f'process-start: {p}')
        p.start()
    
    try:
        # while True :
        #     continue
        while any(p.is_alive() for p in processes):  # 프로세스가 살아있는 동안 루프 실행
            time.sleep(1)
    finally:
        try:
            for p in processes:
                # print('process-join')
                p.join()
        except Exception as e:
            print(f"Error joining process: {e}")
        
        try:
            for shm in rgb_frames_shm_list + mono_frames_shm_list:
                if shm:
                    # print('shm')
                    shm.close()
                    shm.unlink()
        except Exception as e:
            print(f"Error closing or unlinking shared memory: {e}")

