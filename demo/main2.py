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
    print('1')
    #print(sys.version)


    ######

    # rgb_frames_shm_list = [
    #     shared_memory.SharedMemory(
    #         create = True,
    #         size = np.prod((
    #             config.camera.n_views, 
    #             config.camera.image_height, 
    #             config.camera.image_width, 
    #             3
    #         ))
    #         # *np.dtype(np.uint8).itemsize
    #     ) for _ in range(config.buffer.size)
    # ]

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

    print('2')
    rgb_frames_shm_name_list = [
        shm.name for shm in rgb_frames_shm_list
    ]
    # print(rgb_frames_shm_name_list)
    print('3')
    
    # mono_frames_shm_list = [
    #     shared_memory.SharedMemory(
    #         create = True,
    #         size = np.prod((
    #             config.camera.n_views,
    #             config.camera.image_height,
    #             config.camera.image_width
    #         ))
    #     ) for _ in range(config.buffer.size)
    # ]

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

    print('4')
    mono_frames_shm_name_list = [
        shm.name for shm in mono_frames_shm_list
    ]
    # print(mono_frames_shm_name_list)
    print(mono_frames_shm_list)
    print('5')
    
    #####




    stop_event = mp.Event()

    cam2mp = mp.Queue()
    mp2ume = mp.Queue()
    ume2vz = mp.Queue()

    print('queue')

    # cam2mp = mp.Queue()
    # camera_reader = CameraReader(
    #     config = config,
    #     rgb_frames_shm_name_list = rgb_frames_shm_name_list,
    #     mono_frames_shm_name_list = mono_frames_shm_name_list,
    #     cam2mp = cam2mp,
    #     stop_event = stop_event,
    #     verbose = False
    # )
    # # cam2mp.start()
    # print('cam2mp')
    
    # mp2ume = mp.Queue()
    # media_pipe_estimator = MediaPipeEstimator(  
    #     config = config,
    #     rgb_frames_shm_name_list = rgb_frames_shm_name_list,
    #     cam2mp = cam2mp,
    #     mp2ume = mp2ume,
    #     stop_event = stop_event,
    #     verbose = False
    # )
    # # mp2ume.start()
    # print('mp2ume')

    # ume2vz = mp.Queue()
    # ume_tracker = UmeTracker(
    #     config = config,
    #     shm_name_list = mono_frames_shm_name_list,
    #     mp2ume = mp2ume,
    #     ume2vz = ume2vz,
    #     stop_event = stop_event,
    #     verbose = False
    # )
    # # ume_tracker.start()
    # print('ume2vz')

    # image_visualizer = ImageVisualizer(
    #     config = config,
    #     shared_array_rgb_names = rgb_frames_shm_name_list,
    #     ume2imgviz = ume2vz,
    #     stop_event = stop_event,
    #     verbose = False
    # )
    # # image_visualizer.start()
    # print('image_visualizer')

    # # camera_reader.join()
    # # media_pipe_estimator.join()
    # # ume_tracker.join()
    # image_visualizer.join()

    # stop_event = mp.Event()
    
    # camera_reader = CameraReader(
    #     config = config,
    #     rgb_frames_shm_name_list = rgb_frames_shm_name_list,
    #     mono_frames_shm_name_list = mono_frames_shm_name_list,
    #     cam2mp = cam2mp,
    #     stop_event = stop_event,
    #     verbose = False
    # )
    # media_pipe_estimator = MediaPipeEstimator(  
    #     config = config,
    #     rgb_frames_shm_name_list = rgb_frames_shm_name_list,
    #     cam2mp = cam2mp,
    #     mp2ume = mp2ume,
    #     stop_event = stop_event,
    #     verbose = False
    # )
    ume_tracker = UmeTracker(
        config = config,
        shm_name_list = mono_frames_shm_name_list,
        mp2ume = mp2ume,
        ume2vz = ume2vz,
        stop_event = stop_event,
        verbose = False
    )
    # image_visualizer = ImageVisualizer(
    #     config = config,
    #     shared_array_rgb_names = rgb_frames_shm_name_list,
    #     ume2imgviz = ume2vz,
    #     stop_event = stop_event,
    #     verbose = False
    # )


    # processes = [camera_reader, media_pipe_estimator, image_visualizer, ume_tracker] # 이게 가장 잘 동작하는 것 같음
    # processes = [camera_reader, media_pipe_estimator, ume_tracker ,image_visualizer]

    # processes = [camera_reader, image_visualizer]
    # processes = [image_visualizer, ume_tracker, media_pipe_estimator, camera_reader]
    # processes = [camera_reader, media_pipe_estimator, ume_tracker]
    # processes = [camera_reader]
    # processes = [media_pipe_estimator]
    processes = [ume_tracker]
    # processes = [image_visualizer]

    
    # print(processes)

    # camera_reader.start()
    # print(f'process-start: {camera_reader}')
    # media_pipe_estimator.start()
    # print(f'process-start: {media_pipe_estimator}')
    # ume_tracker.start()
    # print(f'process-start: {ume_tracker}')
    # image_visualizer.start()
    # print(f'process-start: {image_visualizer}')
    
    for p in processes:
        # print('process-start')
        print(f'process-start: {p}')
        
        p.start()
    
    # for p in processes:
    #     print('process-join')
    #     p.join()


    # try:
    #     while True :
    #         continue
    # finally:
    #     for p in processes:
    #         print('process-join')
    #         p.join()
    #     for shm in rgb_frames_shm_list + mono_frames_shm_list:
    #         print('shm')
    #         shm.close()
    #         shm.unlink()
    
    # try:
    #     for p in processes:
    #         print('process-join')
    #         p.join()
    # except Exception as e:
    #     print(f'Error joining process : {e}')

    # try:
    #     for shm in rgb_frames_shm_list + mono_frames_shm_list:
    #         print('shm')
    #         shm.close()
    #         shm.unlink()
    # except Exception as e:
    #     print(f"Error closing or unlinking shared memory: {e}")


    print('process--while')
    try:
        # while True :
        #     continue
        while any(p.is_alive() for p in processes):  # 프로세스가 살아있는 동안 루프 실행
            time.sleep(1)
    finally:
        try:
            for p in processes:
                print('process-join')
                p.join()
        except Exception as e:
            print(f"Error joining process: {e}")
        
        try:
            for shm in rgb_frames_shm_list + mono_frames_shm_list:
                if shm:
                    print('shm')
                    shm.close()
                    shm.unlink()
        except Exception as e:
            print(f"Error closing or unlinking shared memory: {e}")

