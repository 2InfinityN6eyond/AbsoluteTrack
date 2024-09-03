import cv2
import mediapipe as mp
import time
import numpy as np


def open_stereo_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
    for CAM_ID in range(-1, CAM_ID_MAX) :
        cap = cv2.VideoCapture(CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
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
    hands_left = mp_hands.Hands(
        max_num_hands = 2,
        min_detection_confidence = 0.5, 
        min_tracking_confidence = 0.5
    )
    # hands_right = mp_hands.Hands(
    #     max_num_hands = 2,
    #     min_detection_confidence = 0.5, 
    #     min_tracking_confidence = 0.5
    # )

    cv2.namedWindow('hand', cv2.WINDOW_NORMAL)


    fps_outer = 0
    fps_inner = 0

    while cap.isOpened():
        stt = time.time()
        
        ret, frame_stereo = cap.read()
        if not ret:
            break
        
        frame_left = frame_stereo[:, :IMAGE_WIDTH]
        frame_right = frame_stereo[:, IMAGE_WIDTH:]

        '''
        
        result_left = hands_left.process(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
        result_right = hands_left.process(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))

        mp_hand_pose_dict_left = {}
        if result_left.multi_handedness :
            for handedness, landmark in zip(
                result_left.multi_handedness,
                result_left.multi_hand_landmarks
            ) :
                hand_index = handedness.classification[0].index
                hand_pose = np.array(list(map(
                    lambda l : [l.x, l.y, l.z],
                    landmark.landmark
                )))
                mp_hand_pose_dict_left[hand_index] = hand_pose
        
        mp_hand_pose_dict_right = {}
        if result_right.multi_handedness :
            for handedness, landmark in zip(
                result_right.multi_handedness,
                result_right.multi_hand_landmarks
            ) :
                hand_index = handedness.classification[0].index
                hand_pose = np.array(list(map(
                    lambda l : [l.x, l.y, l.z],
                    landmark.landmark
                )))
                mp_hand_pose_dict_right[hand_index] = hand_pose


        handness_color_map = {
            0 : (0, 0, 255), # RED
            1 : (0, 255, 0), # GREEN
        }

        fps_inner = 0.5 * fps_inner + 0.5 * (1 / (time.time() - stt))

        for hand_idx, hand_pose in mp_hand_pose_dict_left.items() :
            for point in hand_pose :
                x, y, z = point
                cv2.circle(frame_left, (int(x), int(y)), 5, handness_color_map[hand_idx], -1)

        for hand_idx, hand_pose in mp_hand_pose_dict_right.items() :
            for point in hand_pose :
                x, y, z = point
                cv2.circle(frame_right, (int(x), int(y)), 5, handness_color_map[hand_idx], -1)
        '''

        k = cv2.waitKey(1)
        if k==49:
            break
        cv2.imshow('hand', frame_left)

        fps_outer = 0.5 * fps_outer + 0.5 * (1 / (time.time() - stt))
        print(fps_outer)
        # print(f"FPS: {fps_outer:.2f}, {fps_inner:.2f}")
    cap.release()
    cv2.destroyAllWindows()