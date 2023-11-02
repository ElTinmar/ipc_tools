from multiprocessing import Process, Event
import numpy as np
import time
import cv2

from ring_buffer import RingBuffer, OverflowRingBuffer_Locked

SZ = (2048,2048)
BIGARRAY = np.random.randint(0, 255, SZ, dtype=np.uint8)

def consumer_cv(ring_buf: RingBuffer, stop: Event, sleep_time: float):
    while not stop.is_set():
        array = ring_buf.get()
        if array is not None:
            cv2.imshow('display',array)
            cv2.waitKey(1)
    cv2.destroyAllWindows()

def producer_random(ring_buf: RingBuffer, stop: Event, sleep_time: float):
    while not stop.is_set():
        ring_buf.put(np.random.randint(0, 255, SZ, dtype=np.uint8))

def consumer(ring_buf: RingBuffer, stop: Event, sleep_time: float):
    while not stop.is_set():
        array = ring_buf.get()
        time.sleep(sleep_time)

def producer(ring_buf: RingBuffer, stop: Event, sleep_time: float):
    while not stop.is_set():
        ring_buf.put(BIGARRAY)
        time.sleep(sleep_time)

def monitor(ring_buf: RingBuffer, stop: Event, sleep_time: float):
    while not stop.is_set():
        print(ring_buf.size())
        time.sleep(sleep_time)

def test_00():
    # 1 producer 
    # 1 consumer
    # producer and consumer ~ same speed

    buffer = OverflowRingBuffer_Locked(
        num_items = 100, 
        item_shape = SZ,
        data_type = np.uint8
    )

    stop = Event()

    p0 = Process(target=producer,args=(buffer,stop,0.001))
    p1 = Process(target=consumer,args=(buffer,stop,0.001))
    p2 = Process(target=monitor,args=(buffer,stop,0.001))

    p0.start()
    p1.start()
    p2.start()

    time.sleep(2)
    stop.set()

    p0.join()
    p1.join()
    p2.join()

def test_01():
    # 1 producer 
    # 1 consumer
    # producer faster than consumer

    buffer = OverflowRingBuffer_Locked(
        num_items = 100, 
        item_shape = SZ,
        data_type = np.uint8
    )

    stop = Event()

    p0 = Process(target=producer,args=(buffer,stop,0.001))
    p1 = Process(target=consumer,args=(buffer,stop,0.002))
    p2 = Process(target=monitor,args=(buffer,stop,0.001))

    p0.start()
    p1.start()
    p2.start()

    time.sleep(2)
    stop.set()

    p0.join()
    p1.join()
    p2.join()

def test_02():
    # 1 producer 
    # 1 consumer
    # consumer faster than producer

    buffer = OverflowRingBuffer_Locked(
        num_items = 100, 
        item_shape = SZ,
        data_type = np.uint8
    )

    stop = Event()

    p0 = Process(target=producer,args=(buffer,stop,0.002))
    p1 = Process(target=consumer,args=(buffer,stop,0.001))
    p2 = Process(target=monitor,args=(buffer,stop,0.001))

    p0.start()
    p1.start()
    p2.start()

    time.sleep(2)
    stop.set()

    p0.join()
    p1.join()
    p2.join()

def test_03():
    # 2 producer 
    # 1 consumer

    buffer = OverflowRingBuffer_Locked(
        num_items = 100, 
        item_shape = SZ,
        data_type = np.uint8
    )

    stop = Event()

    p0 = Process(target=producer,args=(buffer,stop,0.001))
    p1 = Process(target=producer,args=(buffer,stop,0.001))
    p2 = Process(target=consumer,args=(buffer,stop,0.001))
    p3 = Process(target=monitor,args=(buffer,stop,0.001))

    p0.start()
    p1.start()
    p2.start()
    p3.start()

    time.sleep(2)
    stop.set()

    p0.join()
    p1.join()
    p2.join()
    p3.join()

def test_04():
    # 1 producer 
    # 2 consumer

    buffer = OverflowRingBuffer_Locked(
        num_items = 100, 
        item_shape = SZ,
        data_type = np.uint8
    )

    stop = Event()

    p0 = Process(target=producer,args=(buffer,stop,0.001))
    p1 = Process(target=consumer,args=(buffer,stop,0.001))
    p2 = Process(target=consumer,args=(buffer,stop,0.001))
    p3 = Process(target=monitor,args=(buffer,stop,0.001))

    p0.start()
    p1.start()
    p2.start()
    p3.start()

    time.sleep(2)
    stop.set()

    p0.join()
    p1.join()
    p2.join()
    p3.join()

def test_05():
    # 1 producer 
    # 1 consumer
    # producer and consumer ~ same speed

    buffer = OverflowRingBuffer_Locked(
        num_items = 100, 
        item_shape = SZ,
        data_type = np.uint8
    )

    stop = Event()

    p0 = Process(target=producer_random,args=(buffer,stop,0.001))
    p1 = Process(target=consumer_cv,args=(buffer,stop,0.001))
    p2 = Process(target=monitor,args=(buffer,stop,0.001))

    p0.start()
    p1.start()
    p2.start()

    time.sleep(2)
    stop.set()

    p0.join()
    p1.join()
    p2.join()

def test_overflow():

    buffer = OverflowRingBuffer_Locked(
        num_items = 5, 
        item_shape = (1,),
        data_type = np.uint8
    )

    buffer.put(np.array(1, dtype=np.uint8))
    buffer.put(np.array(2, dtype=np.uint8))
    buffer.put(np.array(3, dtype=np.uint8))
    buffer.put(np.array(4, dtype=np.uint8))
    buffer.put(np.array(5, dtype=np.uint8))
    buffer.put(np.array(6, dtype=np.uint8))

def test_types():

    buffer = OverflowRingBuffer_Locked(
        num_items = 5, 
        item_shape = (10,),
        data_type = np.float32
    )

    buffer.put(np.ones(shape=(10,), dtype=np.float32))
    

if __name__ == '__main__':
    #test_00()
    #test_01()
    #test_02()
    #test_03()
    #test_04()
    test_05()