import time
import tyro
import cv2
import numpy as np
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import shared_memory


from media_pipe_estimator import MediaPipeEstimator
from ume_tracker import UmeTracker
from image_visualizer import ImageVisualizer




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



class CameraReader(mp.Process):
    def __init__(
        self,
        config,
        rgb_frames_shm_name_list,
        mono_frames_shm_name_list,
        cam2mp,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.rgb_frames_shm_name_list = rgb_frames_shm_name_list
        self.mono_frames_shm_name_list = mono_frames_shm_name_list
        self.cam2mp = cam2mp
        self.stop_event = stop_event
        self.verbose = verbose
        
        
    def run(self):
        self.rgb_frames_shm_list = [
            shared_memory.SharedMemory(
                name = shm_name
            ) for shm_name in self.rgb_frames_shm_name_list
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
            ) for shm in self.rgb_frames_shm_list
        ]

        self.mono_frames_shm_list = [
            shared_memory.SharedMemory(
                name = shm_name
            ) for shm_name in self.mono_frames_shm_name_list
        ]
        self.mono_frames_array_list = [
            np.ndarray(
                shape = (
                    self.config.camera.n_views,
                    self.config.camera.image_height,
                    self.config.camera.image_width
                ),
                dtype = np.uint8,
                buffer = shm.buf
            ) for shm in self.mono_frames_shm_list
        ]
        
        cap = cv2.VideoCapture(self.config.camera.cv_camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.image_width * self.config.camera.n_views)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.image_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        index = 0
        while not self.stop_event.is_set():
            ret, multiview_frame = cap.read()
            if not ret:
                continue

            multiview_frame_rgb = cv2.cvtColor(multiview_frame, cv2.COLOR_BGR2RGB)
            multiview_frame_mono = cv2.cvtColor(multiview_frame, cv2.COLOR_BGR2GRAY)

            for view_idx in range(self.config.camera.n_views):
                self.rgb_frames_array_list[index][view_idx][:] = multiview_frame_rgb[
                    :, 
                    view_idx * self.config.camera.image_width : (view_idx + 1) * self.config.camera.image_width
                ]
                self.mono_frames_array_list[index][view_idx][:] = multiview_frame_mono[
                    :, 
                    view_idx * self.config.camera.image_width : (view_idx + 1) * self.config.camera.image_width
                ]
                
            self.cam2mp.put((index))

            index += 1
            index %= self.config.buffer.size



if __name__ == "__main__":
    config = tyro.cli(Config)

    rgb_frames_shm_list = [
        shared_memory.SharedMemory(
            create = True,
            size = np.prod((
                config.camera.n_views, 
                config.camera.image_height, 
                config.camera.image_width, 
                3
            ))
        ) for _ in range(config.buffer.size)
    ]
    rgb_frames_shm_name_list = [
        shm.name for shm in rgb_frames_shm_list
    ]
    
    mono_frames_shm_list = [
        shared_memory.SharedMemory(
            create = True,
            size = np.prod((
                config.camera.n_views,
                config.camera.image_height,
                config.camera.image_width
            ))
        ) for _ in range(config.buffer.size)
    ]
    mono_frames_shm_name_list = [
        shm.name for shm in mono_frames_shm_list
    ]
    
    cam2mp = mp.Queue()
    mp2ume = mp.Queue()
    ume2vz = mp.Queue()


    stop_event = mp.Event()

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

    processes = [image_visualizer, ume_tracker, media_pipe_estimator, camera_reader]
    

    for p in processes:
        p.start()
    
    try:
        while True :
            continue
    finally:
        for p in processes:
            p.join()
        for shm in rgb_frames_shm_list + mono_frames_shm_list:
            shm.close()
            shm.unlink()
        