import cv2
import mediapipe as mp
import cv2

def open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
    for CAM_ID in range(-1, CAM_ID_MAX) :
        cap = cv2.VideoCapture(CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if cap.isOpened() :
            print(f"Camera ID {CAM_ID} Frame Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            return cap

IMG_SETTING_OPTIONS = {
    "0": (1920, 1080),
    "1": (1280, 720),
    "2": (640, 480),
}
DEFAULT_IMAGE_SETTING = "2"

if __name__ == "__main__":
    IMAGE_SETTING = input(
        f"Enter the image setting (0: {IMG_SETTING_OPTIONS['0']}, 1: {IMG_SETTING_OPTIONS['1']}, 2: {IMG_SETTING_OPTIONS['2']}): "
    ) or DEFAULT_IMAGE_SETTING
    IMAGE_WIDTH, IMAGE_HEIGHT = IMG_SETTING_OPTIONS[IMAGE_SETTING]
    
    CAM_ID_MAX = 10
    cap = open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX)

    # PRETRAIN_PATH = "/home/hjp/HJP/KUAICV/Hand/AbsoluteTrack/agora/hand_landmarker.task"

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    )

    while cap.isOpened():
        ret, frame_stereo = cap.read()
        if not ret:
            break
        
        frame_left = frame_stereo[:, :IMAGE_WIDTH]
        frame_right = frame_stereo[:, IMAGE_WIDTH:]
        
        result = hands.process(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
        
        if not ret: 
            break
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_left,
                    res,
                    mp_hands.HAND_CONNECTIONS
                )
    
        k = cv2.waitKey(1)
        if k==49:
            break
        cv2.imshow('hand', frame_left)
        
    video.release()
    cv2.destroyAllWindows()