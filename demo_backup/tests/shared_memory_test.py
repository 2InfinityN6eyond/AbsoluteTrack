import multiprocessing
import numpy as np
from multiprocessing import shared_memory

def child_process(shm_name, shape, dtype):
    # Attach to the shared memory by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    
    # Recreate the NumPy array backed by the shared memory
    shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    # Read or modify the shared array
    print("Child process reads shared array:", shared_array)
    
    # Perform some operation (for demonstration)
    shared_array[0] = 42  # Example modification
    
    # Close the shared memory in the child process
    existing_shm.close()

if __name__ == "__main__":
    # Create shared memory
    array = np.array([1, 2, 3, 4], dtype=np.int32)
    
    # Create a shared memory block for the array
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    
    # Copy the original array into the shared memory
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array[:]
    
    print("Parent process creates shared array:", shared_array)
    
    # Start the child process, passing the shared memory name, shape, and dtype
    process = multiprocessing.Process(target=child_process, args=(shm.name, array.shape, array.dtype))
    process.start()
    process.join()
    
    # Check the updated value after the child process modifies the shared array
    print("Parent process reads modified shared array:", shared_array)
    
    # Close and unlink the shared memory
    shm.close()
    shm.unlink()