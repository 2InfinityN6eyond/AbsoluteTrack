import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
    for CAM_ID in range(-1, CAM_ID_MAX) :
        cap = cv2.VideoCapture(CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if cap.isOpened() :
            print(f"Camera ID {CAM_ID} Frame Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            return cap

# Create a hand landmarker instance with the live stream mode:
def print_result(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    # print("============== result ==============")
    # print(result)
    # print(type(output_image))
    # We'll use this flag to update the image in the main loop
    global new_image
    global detection_result
    new_image = output_image.numpy_view()
    detection_result = result


IMG_SETTING_OPTIONS = {
    "0": (1920, 1080),
    "1": (1280, 720),
    "2": (640, 480),
}
DEFAULT_IMAGE_SETTING = "2"

if __name__ == "__main__":
    IMAGE_SETTING = input("Enter the image setting (0: 1920x1080, 1: 1280x720, 2: 640x480): ") or DEFAULT_IMAGE_SETTING
    IMAGE_WIDTH, IMAGE_HEIGHT = IMG_SETTING_OPTIONS[IMAGE_SETTING]
    
    CAM_ID_MAX = 10
    cap = open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX)

    PRETRAIN_PATH = "/home/hjp/HJP/KUAICV/Hand/AbsoluteTrack/agora/hand_landmarker.task"

    options = HandLandmarkerOptions(
        base_options    = BaseOptions(model_asset_path=PRETRAIN_PATH),
        running_mode    = VisionRunningMode.LIVE_STREAM,
        result_callback = print_result,
    )
    
    new_image = None  # Initialize new_image
    
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame_stereo = cap.read()
            if not ret:
                break
            
            frame_left = frame_stereo[:, :IMAGE_WIDTH]
            frame_right = frame_stereo[:, IMAGE_WIDTH:]
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_left.copy())
            landmarker.detect_async(mp_image, int(time.time() * 1000))
            
            # Display the original frame
            cv2.imshow("Original", frame_left)
            
            # If we have a new image from the callback, display it
            if new_image is not None and detection_result is not None:
                
                annotated_image = draw_landmarks_on_image(new_image, detection_result)
                cv2.imshow("Hand Landmarks", annotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()