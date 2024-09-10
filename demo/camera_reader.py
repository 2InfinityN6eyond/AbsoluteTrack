import time
import tyro
import cv2
import numpy as np
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import shared_memory

import sys


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
        print('camear-run')
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
        
        # print('camera-camera')
        cap = cv2.VideoCapture(self.config.camera.cv_camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.image_width * self.config.camera.n_views)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.image_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        print('camera-setting')

        index = 0
        while not self.stop_event.is_set():
            # print('camera-start')
            ret, multiview_frame = cap.read()
            if not ret:
                print('ERROR!!! no camera capture!!!')
                continue

            multiview_frame_rgb = cv2.cvtColor(multiview_frame, cv2.COLOR_BGR2RGB)
            multiview_frame_mono = cv2.cvtColor(multiview_frame, cv2.COLOR_BGR2GRAY)

            # print(multiview_frame_rgb)
            # print(multiview_frame_mono)

            # cv2.imshow('camera-reader-rgb', multiview_frame_rgb) # 응답없음 에러남
            # cv2.imshow('camera-reader-mono', multiview_frame_mono) # 응답없음 에러남

            for view_idx in range(self.config.camera.n_views):
                self.rgb_frames_array_list[index][view_idx][:] = multiview_frame_rgb[
                    :, 
                    view_idx * self.config.camera.image_width : (view_idx + 1) * self.config.camera.image_width
                ]
                self.mono_frames_array_list[index][view_idx][:] = multiview_frame_mono[
                    :, 
                    view_idx * self.config.camera.image_width : (view_idx + 1) * self.config.camera.image_width
                ]
                # ex_img = self.mono_frames_array_list[index][view_idx][:]
                # print(ex_img)
                # cv2.imwrite('./ex_img.jpg', ex_img)
                
            self.cam2mp.put((index))

            index += 1
            index %= self.config.buffer.size
            # print(f'cam2mp-{index}')
            # print(self.cam2mp.get())