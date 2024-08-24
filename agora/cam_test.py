import cv2


def open_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID_MAX = 10) :
    for CAM_ID in range(-1, CAM_ID_MAX) :
        cap = cv2.VideoCapture(CAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if cap.isOpened() :
            print(f"Camera ID {CAM_ID} Frame Width {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            return cap

IMG_SETTING_OPTIONS = {
    "0": (2560, 960),
    "1": (1280, 720),
    "2": (640, 480),
}
 
IMAGE_WIDTH, IMAGE_HEIGHT = IMG_SETTING_OPTIONS[input(
    f"Enter the image setting (0: {IMG_SETTING_OPTIONS['0']}, 1: {IMG_SETTING_OPTIONS['1']}, 2: {IMG_SETTING_OPTIONS['2']}): "
)]

print(IMAGE_WIDTH, IMAGE_HEIGHT)
CAM_ID = 1

cap = open_camera(IMAGE_WIDTH, IMAGE_HEIGHT, CAM_ID)

while True:
    ret, frame = cap.read()
    print(frame.shape)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break