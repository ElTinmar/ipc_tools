import time
from  multiprocessing import Event, Queue, Process
import numpy as np
import matplotlib.pyplot as plt
from ipc_tools import ModifiableRingBuffer 
from multiprocessing.synchronize import Event as EventType
import sys

NUM_PRODUCERS = 1
NUM_CONSUMERS = 1
BUFFER_SIZE_BYTES = 500 * 1024**2
ITEM_SHAPE = (2048, 2048)
DTYPE = np.float32
RUNTIME_SEC = 10
LOGGING = False
ITEM_SIZE_BYTES = np.prod(ITEM_SHAPE)*np.dtype(DTYPE).itemsize

def producer(buffer: ModifiableRingBuffer, stop: EventType, stats: Queue):
    times = []
    item = np.random.rand(*ITEM_SHAPE).astype(DTYPE)
    while not stop.is_set():
        try:
            t0 = time.perf_counter()
            buffer.put(item)
            if sys.platform == "linux":
                time.sleep(1e-6) # leave some time for consumer
            t1 = time.perf_counter()
            times.append(t1 - t0)
        except Exception as e:
            continue

    stats.put(('put', times))

def consumer(buffer: ModifiableRingBuffer, stop: EventType, stats: Queue):
    times = []
    while not stop.is_set():
        try:
            t0 = time.perf_counter()
            item = buffer.get(block=True, timeout=1.0)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        except Exception as e:
            continue

    stats.put(('get', times))

def run_test():
    stop_event = Event()
    stats_queue = Queue()

    buffer = ModifiableRingBuffer(
        num_bytes=BUFFER_SIZE_BYTES,
        copy=False,
        name='benchmark_buffer',
    )

    producers = [Process(target=producer, args=(buffer, stop_event, stats_queue)) for _ in range(NUM_PRODUCERS)]
    consumers = [Process(target=consumer, args=(buffer, stop_event, stats_queue)) for _ in range(NUM_CONSUMERS)]

    for p in producers + consumers:
        p.start()

    print(f"Running benchmark for {RUNTIME_SEC} seconds...")
    time.sleep(RUNTIME_SEC)
    stop_event.set()

    time.sleep(10)

    # Collect stats
    put_times = []
    get_times = []
    while not stats_queue.empty():
        kind, times = stats_queue.get()
        if kind == 'put':
            put_times.extend(times)
        elif kind == 'get':
            get_times.extend(times)

    for p in producers + consumers:
        p.join()

    print(f"Done")
    print(stats_queue.qsize())

    def summarize(label, times):
        times = np.array(times)
        print(f"{label}:")
        print(f"  Count: {len(times)}")
        print(f"  Mean:  {np.mean(times)*1000:.3f} ms")
        print(f"  Std:   {np.std(times)*1000:.3f} ms")
        print(f"  Rate:  {len(times)/RUNTIME_SEC:.2f} ops/sec\n")
        print(f"  Throughput:  {1e-6* (len(times)*ITEM_SIZE_BYTES)/RUNTIME_SEC:.2f} MB/s\n")

    print("\n--- Performance Summary ---")
    summarize("Producer put()", put_times)
    summarize("Consumer get()", get_times)

    # Optional plot
    plt.hist(np.array(put_times)*1000, bins=100, alpha=0.7, label='put()')
    plt.hist(np.array(get_times)*1000, bins=100, alpha=0.7, label='get()')
    plt.xlabel("Op Time (ms)")
    plt.ylabel("Count")
    plt.title("Ring Buffer Operation Latencies")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_test()