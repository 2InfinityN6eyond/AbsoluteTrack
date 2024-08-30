import multiprocessing as mp
from multiprocessing import Manager
import time

def hand_pose_estimator(shared_mp_pose_dict_queue, stop_event):
    while not stop_event.is_set():
        # Simulate some work
        hand_pose_results_dict = {0: {0: [1, 2, 3], 1: [4, 5, 6]}}
        shared_mp_pose_dict_queue[0] = hand_pose_results_dict
        #time.sleep(1)  # Simulate processing time

def hand_tracker_3d(shared_mp_pose_dict_queue, stop_event):
    while not stop_event.is_set():
        if shared_mp_pose_dict_queue[0]:
            print("Received:", shared_mp_pose_dict_queue[0])
        #time.sleep(1)

if __name__ == "__main__":
    manager = Manager()
    shared_mp_pose_dict_queue = manager.list([manager.dict()])

    stop_event = mp.Event()

    hand_pose_estimator_process = mp.Process(
        target=hand_pose_estimator,
        args=(shared_mp_pose_dict_queue, stop_event)
    )
    hand_tracker_3d_process = mp.Process(
        target=hand_tracker_3d,
        args=(shared_mp_pose_dict_queue, stop_event)
    )

    hand_pose_estimator_process.start()
    hand_tracker_3d_process.start()

    try:
        time.sleep(10)  # Run for 10 seconds
    finally:
        stop_event.set()
        hand_pose_estimator_process.join()
        hand_tracker_3d_process.join()