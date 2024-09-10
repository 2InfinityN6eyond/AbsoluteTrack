import numpy as np
from multiprocessing import shared_memory

# 공유 메모리 생성 (서버 측)
def create_shared_memory():
    # 공유할 데이터 배열 생성
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    
    # 데이터 크기에 맞게 공유 메모리 생성
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)

    # 공유 메모리에 데이터 복사
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    
    print("Shared memory created with name:", shm.name)
    return shm

# 공유 메모리 사용 (클라이언트 측)
def access_shared_memory(shm_name):
    # 공유 메모리에 연결
    existing_shm = shared_memory.SharedMemory(name=shm_name)

    # 공유 메모리에서 데이터를 읽음
    shared_array = np.ndarray((5,), dtype=np.int64, buffer=existing_shm.buf)
    print("Data in shared memory:", shared_array)

    # 공유 메모리 사용이 끝나면 닫기
    existing_shm.close()

# 공유 메모리 해제
def close_shared_memory(shm):
    shm.close()  # 메모리 객체 닫기
    shm.unlink()  # 메모리 해제

# 실행
if __name__ == "__main__":
    # 공유 메모리 생성
    shm = create_shared_memory()

    # 공유 메모리 접근 (다른 프로세스나 스레드에서 이 함수를 호출한다고 가정)
    access_shared_memory(shm.name)

    # 공유 메모리 해제
    close_shared_memory(shm)
