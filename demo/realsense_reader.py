import time
import numpy as np
import cv2
import pyrealsense2 as rs
import multiprocessing as mp

class RealsenseReader(mp.Process) :
    def __init__(
        self,
        config,
        depth_shm_name_list,
        stop_event,
        verbose = False
    ) :
        super().__init__()
        
        self.config = config
        self.depth_shm_name_list = depth_shm_name_list
        self.stop_event = stop_event
        self.verbose = verbose
        
        
    def run(self) :
        # open realsense stream
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.config.rs.width, self.config.rs.height,
            rs.format.bgr8, self.config.rs.fps
        )
        self.rs_config.enable_stream(
            rs.stream.depth, self.config.rs.width, self.config.rs.height,
            rs.format.z16, self.config.rs.fps
        )
        # self.rs_config.enable_stream(
        #     rs.stream.infrared, 1, self.config.rs.width, self.config.rs.height,
        #     rs.format.y8, self.config.rs.fps
        # )
        # self.rs_config.enable_stream(rs.stream.infrared, 2, self.config.width, self.config.height, rs.format.y8, self.config.fps)
        
        profile = self.rs_pipeline.start(self.rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale is: {depth_scale}")
        
        clipping_distance_in_meters = 2
        clipping_distance = clipping_distance_in_meters / depth_scale
        
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        while not self.stop_event.is_set() :
            frames = self.rs_pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            colored_depth_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
        
            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", colored_depth_image)
            key = cv2.waitKey(1)
            if key == ord("q") :
                break
        
        self.rs_pipeline.stop()
        cv2.destroyAllWindows()
        
        
        
        
if __name__ == "__main__" :    
    from dataclasses import dataclass

    @dataclass
    class RealsenseConfig :
        width : int = 640
        height : int = 480
        fps : int = 30
        
    class Config :
        rs : RealsenseConfig = RealsenseConfig()
        
        
    stop_event = mp.Event()
        
    realsense_reader = RealsenseReader(
        config = Config(),
        depth_shm_name_list = [],
        stop_event = mp.Event(),
        verbose = True
    )
        
    realsense_reader.start()
    
    while not stop_event.is_set() :
        time.sleep(1)
        
    realsense_reader.join()
        