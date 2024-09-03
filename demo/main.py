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
        camreader2mp,
        stop_event,
        verbose = False
    ):
        super().__init__()
        self.config = config
        self.shared_array_rgb = shared_array_rgb
        self.shared_array_mono = shared_array_mono
        self.camreader2mp = camreader2mp
        self.stop_event = stop_event
        self.verbose = verbose
        
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

        fps1 = 0
        index = 0
        while not self.stop_event.is_set():
            stt1 = time.time()
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
    
            self.camreader2mp.put(index)
                
            index += 1
            index %= len(self.shared_array_rgb)
            
            fps1 = 0.5 * fps1 + 0.5 * (1 / (time.time() - stt1))
            
            if self.verbose:
                print(f"CAM FPS: {int(fps1)}")
            
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
        
        mp_handedness_color_map = {
            0: (255, 100, 0), # red
            1: (100, 255, 100), # green
        }
        
        ume_handedness_color_map = {
            0: (0, 0, 255), # red
            1: (0, 255, 0), # green
        }
        
        
        fps1 = 0
        fps2 = 0
        while not self.stop_event.is_set():
            stt1 = time.time()
        
            (
                index, 
                mp_hand_pose_dict, 
                tracked_keypoints_dict,
                projected_keypoints_dict
            ) = self.ume2imgviz.get()
            
            stt2 = time.time()
            
            frame = self.shared_array_rgb_list[index]
            mp_pose_results = self.shared_mp_pose_list[index]
            
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
                            x, y, z = keypoint
                            cv2.circle(
                                img, (int(x), int(y)), 5, 
                                ume_handedness_color_map[hand_idx], -1
                            )
                
                
                
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
    
    camreader2mp = mp.Queue()
    mp2ume = mp.Queue()
    ume2imgviz = mp.Queue()
    
    camera_reader = CameraReader(
        config              = config, 
        shared_array_rgb    = shared_array_rgb,
        shared_array_mono   = shared_array_mono,
        camreader2mp        = camreader2mp,
        stop_event          = stop_event,
        verbose             = False
    )
    media_pipe_estimator = MediaPipeEstimator(
        config                  = config,
        shared_array_rgb        = shared_array_rgb,
        shared_mp_pose_array    = shared_mp_pose_array,
        camreader2mp            = camreader2mp,
        mp2ume                  = mp2ume,
        stop_event              = stop_event,
        verbose                 = False
    )
    ume_tracker = UmeTracker(
        config                  = config, 
        shared_array_mono       = shared_array_mono,
        shared_mp_pose_array    = shared_mp_pose_array,
        shared_ume_pose_array   = shared_ume_pose_array,
        mp2ume                  = mp2ume,
        ume2imgviz              = ume2imgviz,
        stop_event              = stop_event,
        verbose                 = False
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
        media_pipe_estimator,
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