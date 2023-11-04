from ring_buffer import OverflowRingBuffer_Locked
from multiprocessing import  Value, queues, get_context
from typing import Optional
from numpy.typing import NDArray, ArrayLike
import time


class MonitoredRingBuffer(OverflowRingBuffer_Locked):

    def __init__(self, refresh_every: int = 100, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.num_item_in = Value('I',0)
        self.num_item_out = Value('I',0)
        self.time_in = Value('d',0)
        self.time_out = Value('d',0)
        
        self.refresh_every = refresh_every

    def put(self, element: ArrayLike) -> None:
        super().put(element)
        with self.num_item_in.get_lock():
            self.num_item_in.value += 1
        self.display_put()

    def get(self, blocking: bool = True, timeout: float = float('inf')) -> Optional[NDArray]:
        res = super().get(blocking, timeout)
        with self.num_item_out.get_lock():
            self.num_item_out.value += 1
        self.display_get()        

    def display_get(self) -> None:

        if (self.num_item_out.value % self.refresh_every == 0):
            previous_time = self.time_out.value
            self.time_out.value = time.monotonic()
            fps = self.refresh_every/(self.time_out.value - previous_time)
            print(f'FPS out: {fps}, buffer size: {self.size()}')

    def display_put(self) -> None:
        
        if (self.num_item_in.value % self.refresh_every == 0):
            previous_time = self.time_in.value
            self.time_in.value = time.monotonic()
            fps = self.refresh_every/(self.time_in.value - previous_time)
            print(f'FPS in: {fps}, buffer size: {self.size()}')


class MonitoredQueue(queues.Queue):

    def __init__(self, refresh_every: int = 100, *args, **kwargs):

        ctx = get_context()
        super().__init__(*args, **kwargs, ctx=ctx)

        self.num_item_in = Value('I',0)
        self.num_item_out = Value('I',0)
        self.time_in = Value('d',0)
        self.time_out = Value('d',0)
        
        self.refresh_every = refresh_every

    def put(self, element: ArrayLike) -> None:
        super().put(element)
        with self.num_item_in.get_lock():
            self.num_item_in.value += 1
        self.display_put()

    def get(self, blocking: bool = True, timeout: float = float('inf')) -> Optional[NDArray]:
        res = super().get(blocking, timeout)
        with self.num_item_out.get_lock():
            self.num_item_out.value += 1
        self.display_get()        

    def display_get(self):

        if (self.num_item_out.value % self.refresh_every == 0):
            previous_time = self.time_out.value
            self.time_out.value = time.monotonic()
            fps = self.refresh_every/(self.time_out.value - previous_time)
            print(f'FPS out: {fps}, buffer size: {self.qsize()}')

    def display_put(self):
        
        if (self.num_item_in.value % self.refresh_every == 0):
            previous_time = self.time_in.value
            self.time_in.value = time.monotonic()
            fps = self.refresh_every/(self.time_in.value - previous_time)
            print(f'FPS in: {fps}, buffer size: {self.qsize()}')

