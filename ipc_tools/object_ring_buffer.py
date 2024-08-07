from .queue_like import QueueLike
from typing import Optional, Any, Callable
from numpy.typing import NDArray, ArrayLike, DTypeLike
from ipc_tools import RingBuffer
import numpy as np
from multiprocessing_logger import Logger
import time
from queue import Empty
 
class ObjectRingBuffer(QueueLike):

    def __init__(
            self, 
            serialize: Callable[[Any], NDArray], 
            deserialize: Callable[[NDArray], Any],
            data_type: DTypeLike,
            item_shape: ArrayLike = (1,),
            num_items: int = 100,
            t_refresh: float = 1e-6,
            copy: bool = False,
            name: str = '', 
            logger: Optional[Logger] = None,
        ) -> None:

        super().__init__()

        self.num_items = num_items
        self.t_refresh = t_refresh
        self.item_shape = item_shape
        self.data_type = np.dtype(data_type)
        self.copy = copy
        self.name = name

        # create default RingBuffer
        self.queue = RingBuffer(
            num_items = num_items,
            item_shape = item_shape,
            data_type = data_type,
            t_refresh = t_refresh,
            copy = copy,
            logger = logger,
            name = name
        )

        self.serialize = serialize
        self.deserialize = deserialize

    def qsize(self) -> int:
        return self.queue.qsize()

    def empty(self) -> bool:
        return self.queue.empty()
    
    def full(self) -> bool:
        return self.queue.full()

    def put(self, obj: Any, block: Optional[bool] = True, timeout: Optional[float] = None) -> None:
        array = self.serialize(obj) 
        self.queue.put(array, block, timeout)
    
    def get(self, block: Optional[bool] = True, timeout: Optional[float] = None) -> Any:
        array = self.queue.get(block, timeout)
        obj = self.deserialize(array)
        return obj

    def close(self) -> None:
        self.queue.close()

    def join_thread(self) -> None:
        self.queue.join_thread()

    def cancel_join_thread(self) -> None:
        self.queue.cancel_join_thread()

    @property
    def num_lost_item(self):
        return self.queue.num_lost_item

    def clear(self) -> None:
        self.queue.clear()


class ObjectRingBuffer2(RingBuffer):

    def __init__(
            self, 
            serialize: Callable[[NDArray, Any], None], 
            deserialize: Callable[[NDArray], Any],
            *args, **kwargs
        ) -> None:

        super().__init__(*args,**kwargs)

        self.serialize = serialize
        self.deserialize = deserialize

    def get_noblock(self) -> Optional[NDArray]:
        '''return data at the current read location'''
        
        t_start = time.perf_counter_ns() * 1e-6

        with self.lock:

            t_lock_acquired = time.perf_counter_ns() * 1e-6

            if self.empty():
                raise Empty

            if self.copy:
                element = np.frombuffer(
                    self.data, 
                    dtype = self.element_type, 
                    count = self.item_num_element,
                    offset = self.read_cursor.value * self.item_num_element * self.element_byte_size # offset should be in bytes
                ).copy()
            else:
                element = np.frombuffer(
                    self.data, 
                    dtype = self.element_type, 
                    count = self.item_num_element,
                    offset = self.read_cursor.value * self.item_num_element * self.element_byte_size # offset should be in bytes
                )
            self.read_cursor.value = (self.read_cursor.value  +  1) % self.num_items

            t_lock_released = time.perf_counter_ns() * 1e-6

            element = element.reshape(self.item_shape)

            data = self.deserialize(element)

        if self.local_logger:
            self.local_logger.info(f'get, {t_start}, {t_lock_acquired}, {t_lock_released}')

        # this seems to be necessary to give time to other workers to get the lock 
        time.sleep(self.t_refresh)

        return data
    
    def put(self, data: Any, block: Optional[bool] = True, timeout: Optional[float] = None) -> None:
        '''
        Return data at the current write location.
        block and timeout are there for compatibility with the Queue interface, but 
        are ignored since the ring buffer overflows by design.  
        '''

        t_start = time.perf_counter_ns() * 1e-6

        with self.lock:

            t_lock_acquired = time.perf_counter_ns() * 1e-6

            buffer = np.frombuffer(
                self.data, 
                dtype = self.element_type, 
                count = self.item_num_element,
                offset = self.write_cursor.value * self.item_num_element * self.element_byte_size # offset should be in bytes
            )
            buffer = buffer.reshape(self.item_shape)

            # if the buffer is full, overwrite the next block
            if self.full():
                self.read_cursor.value = (self.read_cursor.value  +  1) % self.num_items
                self.num_lost_item.value += 1

            # serialize your data directly into the buffer (avoids extra copy)
            self.serialize(buffer, data)

            # update write cursor value
            self.write_cursor.value = (self.write_cursor.value  +  1) % self.num_items

            t_lock_released = time.perf_counter_ns() * 1e-6

        if self.local_logger:
            self.local_logger.info(f'put, {t_start}, {t_lock_acquired}, {t_lock_released}')

        # this seems to be necessary to give time to other workers to get the lock 
        time.sleep(self.t_refresh)
