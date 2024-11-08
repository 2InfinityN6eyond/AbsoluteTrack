import time
import multiprocessing as mp

import numpy as np
import cv2

import leap
from leap.enums import PolicyFlag, HandType
from leapc_cffi import ffi, libleapc

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

# Leap and UmeTrack has different joint numbering
LEAP2UME_JOINT_MAP = [
    4, 8, 12, 16, 20, # end of fingers
    0, # wrist
    2, 3, # thumb
    5, 6, 7, # index
    9, 10, 11, # middle
    13, 14, 15, # ring
    17, 18, 19, # pinky
    1, # palm
]

class MyListener(leap.Listener):
    def __init__(self):
        super().__init__()
        self.latest_frame = None
        self.latest_images = None
        self.hand_positions = {0: np.zeros((21, 3)), 1: np.zeros((21, 3))}
        
    def get_joint_position(self, bone):
        if bone :
            return bone.x, bone.y, bone.z
        else:
            return None

    def on_tracking_event(self, event):
        self.latest_frame = event
        self.hand_positions = {0: None, 1: None}

        hand_array = np.zeros((2, 21, 3))

        for hand_idx in range(0, len(event.hands)):
            hand = event.hands[hand_idx]
          
            hand_arr_idx = 0 if hand.type == HandType.Left else 1
            
            joint_idx = 0
            wrist = self.get_joint_position(hand.arm.next_joint)
            if wrist:
                hand_array[hand_arr_idx, joint_idx, :] = wrist
                joint_idx += 1
            
            for digit_idx in range(0, 5):
                digit = hand.digits[digit_idx]
                for bone_idx in range(0, 4):
                    bone = digit.bones[bone_idx]

                    
                    # bone_start = self.get_joint_position(bone.prev_joint)
                    bone_end = self.get_joint_position(bone.next_joint)
                    if bone_end :
                        hand_array[hand_arr_idx, joint_idx, :] = bone_end
                        joint_idx += 1


        # print(hand_array)

        self.hand_positions[0] = hand_array[0][LEAP2UME_JOINT_MAP, :].copy()
        self.hand_positions[1] = hand_array[1][LEAP2UME_JOINT_MAP, :].copy()

    def on_image_event(self, event):
        try:
            if hasattr(event, 'image'):
                left_image = event.image[0]
                right_image = event.image[1]
                
                width = left_image.c_data.properties.width
                height = left_image.c_data.properties.height
                
                # Convert image data to numpy array
                left_buffer = ffi.buffer(left_image.c_data.data, width * height)
                right_buffer = ffi.buffer(right_image.c_data.data, width * height)
                
                left_array = np.frombuffer(left_buffer, dtype=np.uint8).reshape(height, width)
                right_array = np.frombuffer(right_buffer, dtype=np.uint8).reshape(height, width)
                
                self.latest_images = (left_array, right_array)
                
        except Exception as e:
            print(f"Error in on_image_event: {e}")



class LeapBridge(mp.Process) :
    
    def __init__(
        self,
        config,
        lp2vz,
        stop_event,
        verbose = False
    ) :
        super().__init__()
        
        self.config = config
        self.lp2vz = lp2vz
        self.stop_event = stop_event
        self.verbose = verbose

    def run(self) :
        self.listener = MyListener()
        connection = leap.Connection()
        connection.add_listener(self.listener)
        
        FLIP_X = True
        FLIP_Y = False    

        with connection.open() :
            connection.set_policy_flags([PolicyFlag.Images])
            connection.set_tracking_mode(leap.TrackingMode.HMD)
            
            
            
            while not self.stop_event.is_set() :
                if self.listener.latest_frame and self.listener.latest_images :
                    tracked_keypoints_dict = {}
                    
                    if self.listener.hand_positions[0] is None and self.listener.hand_positions[1] is None :
                        continue
                    
                    try :
                        for hand_idx, hand_pos in self.listener.hand_positions.items() :
                            if hand_pos is None :
                                continue
                        
                            data = hand_pos.copy()
                        
                            data[:, 1] = hand_pos[:, 2].copy() 
                            data[:, 2] = hand_pos[:, 1].copy()
                            
                            if FLIP_X :
                                data[:, 0] *= -1
                            if FLIP_Y :
                                data[:, 1] *= -1
                            tracked_keypoints_dict[hand_idx] = data.copy()
                    
                        if len(tracked_keypoints_dict) == 2 :
                            # print("sending")
                            # print(tracked_keypoints_dict)
                            try :
                                time.sleep(0.001)  # 1ms delay
                                self.lp2vz.put(tracked_keypoints_dict, timeout=0.01)
                            except mp.queues.Full:
                                continue
                    except Exception as e:
                        print("LEAP BRIDGE ERROR")
                        print(e)

def main():
    import socket
    
    listener = MyListener()
    connection = leap.Connection()
    connection.add_listener(listener)


    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)


    send_period = 10
    with connection.open():
        connection.set_policy_flags([PolicyFlag.Images])
        connection.set_tracking_mode(leap.TrackingMode.HMD)
        
        index = 0
        
        while True:
            if listener.latest_frame and listener.latest_images:
                
                # Display stereo images
                left_image, right_image = listener.latest_images
                cv2.imshow("Left Image", left_image)
                cv2.imshow("Right Image", right_image)

                try : # listener.hand_positions["left"] is not None and listener.hand_positions["right"] is not None:
                    
                    if index == 0 :
                        content = ["L"]
                        for hand_idx in listener.hand_positions.keys() :
                            data = listener.hand_positions[hand_idx].copy()
                            
                            data[:, 1] = listener.hand_positions[hand_idx][:, 2].copy() 
                            data[:, 2] = listener.hand_positions[hand_idx][:, 1].copy() 
                            
                            FLIP_X = True
                            FLIP_Y = True
                            if FLIP_X :
                                data[:, 0] *= -1
                            if FLIP_Y :
                                data[:, 1] *= -1
                                
                            
                            print(data)
                            
                            content.append(str(data.flatten().astype(int).tolist()))
                            
                        content = ";".join(content)
                        sock.sendto(str.encode(str(content)), serverAddressPort)
                        
                    index += 1
                    index %= send_period
                
                except Exception as e:
                    print(e)
                        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()