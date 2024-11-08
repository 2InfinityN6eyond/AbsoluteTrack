import sys
import time
import leap  # Import the Leap Motion SDK

class SampleListener(leap.Listener):
    def on_connect(self, controller):
        print("Connected to Leap Motion")

    def on_frame(self, controller):
        # Get the most recent frame
        frame = controller.frame()

        # Check if any hands are detected
        if not frame.hands.is_empty:
            print(f"Frame id: {frame.id}, Timestamp: {frame.timestamp}, Hands: {len(frame.hands)}")

            # Iterate through hands and extract hand and finger data
            for hand in frame.hands:
                # Determine if the hand is left or right
                hand_type = "Left hand" if hand.is_left else "Right hand"
                print(f"{hand_type} detected with id {hand.id}")

                # Get the hand's position (palm position in 3D space)
                palm_position = hand.palm_position
                print(f"Palm position: {palm_position}")

                # Get the hand's direction and normal vector
                direction = hand.direction
                normal = hand.palm_normal
                print(f"Hand direction: {direction}, Palm normal: {normal}")

                # Get the wrist and arm data
                arm = hand.arm
                print(f"Arm direction: {arm.direction}, Wrist position: {arm.wrist_position}")

                # Get finger data
                for finger in hand.fingers:
                    print(f"  {finger.type_string}: Finger id {finger.id}, Length: {finger.length}mm, Width: {finger.width}mm")

                    # Get the fingertip position
                    fingertip_position = finger.tip_position
                    print(f"    Fingertip position: {fingertip_position}")
                    
                    # Get the direction of the finger
                    finger_direction = finger.direction
                    print(f"    Finger direction: {finger_direction}")

        time.sleep(0.05)  # Slow down the output for clarity (optional)


def main():
    # Create a Leap Motion controller object
    listener = SampleListener()
    controller = leap.Controller()

    # Attach the listener to the controller to receive data
    controller.add_listener(listener)

    print("Press Enter to quit...")
    try:
        sys.stdin.read()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the listener and disconnect the controller
        controller.remove_listener(listener)

if __name__ == "__main__":
    main()
