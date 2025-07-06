from multiprocessing import RawArray, RawValue, RLock, Semaphore
from .queue_like import QueueLike
from typing import Optional
import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike
import time
from queue import Empty
from multiprocessing_logger import Logger
import pickle
import ctypes

class RingBuffer(QueueLike):
    '''
    Simple circular buffer implementation, with the following features:
    - when the buffer is full it will overwrite unread content (overflow)
    - trying to get item from empty buffer can be either blocking (default) or non blocking (return None)
    - supports multiple producers / consumers
    - only one process can access the buffer at a time, writing and reading share the same lock
    - to send multiple fields with heterogeneous type, one can use numpy's structured arrays
    - elements have a fixed size
    '''

    def __init__(
            self,
            num_items: int,
            data_type: DTypeLike,
            item_shape: ArrayLike = (1,),
            copy: bool = False,
            name: str = '',
            logger: Optional[Logger] = None
        ):
        
        self.item_shape = np.asarray(item_shape)
        self.element_type = np.dtype(data_type)
        self.copy = copy
        self.name = name
        self.logger = logger
        self.local_logger = None
        if self.logger:
            self.local_logger = self.logger.get_logger(self.name)
            
        # account for empty slot
        self.num_items = num_items + 1 
        self.item_num_element = int(np.prod(self.item_shape))
        self.element_byte_size = self.element_type.itemsize 
        self.total_size =  self.item_num_element * self.num_items
        
        self.lock = RLock()
        self.semaphore = Semaphore(0)
        self.read_cursor = RawValue('I',0)
        self.write_cursor = RawValue('I',0)
        self.num_lost_item = RawValue('I',0)
        self.data = RawArray('B', self.total_size*self.element_byte_size) 
        
    def get(self, block: bool = True, timeout: Optional[float] = None) -> NDArray:
        '''return buffer to the current read location'''

        if block:
            acquired = self.semaphore.acquire(timeout=timeout)
            if not acquired:
                raise Empty
        else:
            if not self.semaphore.acquire(block=False):
                raise Empty
            
        return self._get_noblock()

    def _get_noblock(self) -> Optional[NDArray]:
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

        if self.local_logger:
            self.local_logger.info(f'get, {t_start}, {t_lock_acquired}, {t_lock_released}')

        return element.reshape(self.item_shape)
    
    def put(self, element: ArrayLike, block: Optional[bool] = True, timeout: Optional[float] = None) -> None:
        '''
        Return data at the current write location.
        block and timeout are there for compatibility with the Queue interface, but 
        are ignored since the ring buffer overflows by design.  
        '''

        t_start = time.perf_counter_ns() * 1e-6

        # convert to numpy array
        arr_element = np.asarray(element, dtype = self.element_type)

        with self.lock:

            t_lock_acquired = time.perf_counter_ns() * 1e-6

            buffer = np.frombuffer(
                self.data, 
                dtype = self.element_type, 
                count = self.item_num_element,
                offset = self.write_cursor.value * self.item_num_element * self.element_byte_size # offset should be in bytes
            )

            # if the buffer is full, overwrite the next block
            if self.full():
                self.read_cursor.value = (self.read_cursor.value  +  1) % self.num_items
                self.num_lost_item.value += 1

            # write flattened array content to buffer (a copy is made)
            buffer[:] = arr_element.ravel()

            # update write cursor value
            self.write_cursor.value = (self.write_cursor.value  +  1) % self.num_items

        self.semaphore.release()
        t_lock_released = time.perf_counter_ns() * 1e-6

        if self.local_logger:
            self.local_logger.info(f'put, {t_start}, {t_lock_acquired}, {t_lock_released}')

    def full(self):
        ''' check if buffer is full '''
        return self.write_cursor.value == ((self.read_cursor.value - 1) % self.num_items)

    def empty(self):
        ''' check if buffer is empty '''
        return self.write_cursor.value == self.read_cursor.value

    def qsize(self):
        ''' Return number of items currently stored in the buffer '''
        return (self.write_cursor.value - self.read_cursor.value) % self.num_items
    
    def close(self):
        pass

    def clear(self):
        '''clear the buffer'''
        with self.lock:
            self.write_cursor.value = self.read_cursor.value

    def get_num_items(self) -> int:
        return self.num_items
    
    def view_data(self):
        num_items = self.write_cursor.value - self.read_cursor.value
        num_element_stored = self.item_num_element * num_items

        stored_data = np.frombuffer(                
            self.data, 
            dtype = self.element_type, 
            count = num_element_stored,
            offset = self.read_cursor.value * self.item_num_element * self.element_byte_size # offset should be in bytes
        ).reshape(np.concatenate(((num_items,) , self.item_shape)))

        return stored_data
    
    def __str__(self):
        
        reprstr = (
            f'capacity: {self.num_items - 1}\n' +
            f'item shape: {self.item_shape}\n' +
            f'data type: {self.element_type}\n' +
            f'size: {self.qsize()}\n' +
            f'read cursor position: {self.read_cursor.value}\n' + 
            f'write cursor position: {self.write_cursor.value}\n' +
            f'lost item: {self.num_lost_item.value}\n' +
            f'buffer: {self.data}\n' + 
            f'{self.view_data()}\n'
        )

        return reprstr
        

class ModifiableRingBuffer(QueueLike):
    '''
    Simple circular buffer implementation, with the following features:
    - when the buffer is full it will overwrite unread content (overflow)
    - trying to get item from empty buffer can be either blocking (default) or non blocking (return None)
    - only one process can access the buffer at a time, writing and reading share the same lock
    - supports multiple producers / consumers
    - to send multiple fields with heterogeneous type, one can use numpy's structured arrays
    - elements can be reshaped, buffer is reinitialized if array dtype changes
    '''

    DTYPE_ARRAY_LEN = 1024*1024
    SHAPE_ARRAY_LEN = 128

    def __init__(
            self,
            num_bytes: int,
            copy: bool = False,
            name: str = '',
            logger: Optional[Logger] = None
        ):
        
        self.num_bytes = num_bytes
        self.copy = copy
        self.name = name
        self.logger = logger
        self.local_logger = None
        if self.logger:
            self.local_logger = self.logger.get_logger(self.name)

        # used to share the data
        self.lock = RLock()
        self.semaphore = Semaphore(0)
        self.read_cursor = RawValue('I',0)
        self.write_cursor = RawValue('I',0)
        self.num_lost_item = RawValue('I',0)
        self.data = RawArray('B', self.num_bytes)

        # this is used to share numpy dtype via pickling
        self.dtype_str_array = RawArray(ctypes.c_char, self.DTYPE_ARRAY_LEN) 
        self.dtype_str_len = RawValue('I',0)

        # share array shape
        self.shape_array = RawArray('Q', self.SHAPE_ARRAY_LEN)
        self.shape_array_len = RawValue('I',0)

        self.element_type = None
        self.element_shape = None
        self.element_byte_size = None
        self.num_items = None
        self.dead_bytes = None
        self.item_shape_product = None

    def load_array_metadata(self):
        
        if self.dtype_str_len.value > 0:
            
            # get dtype
            datatype = pickle.loads(self.dtype_str_array[0:self.dtype_str_len.value])
            if datatype != self.element_type:
                self.element_type = datatype

            shape = np.asarray(self.shape_array[0:self.shape_array_len.value])
            if not np.array_equal(shape, self.element_shape):
                self.element_shape = shape

            self.element_byte_size = self.element_type.itemsize 
            self.item_shape_product = int(np.prod(self.element_shape))
            self.num_items = self.num_bytes // (self.item_shape_product * self.element_byte_size)
            self.dead_bytes = self.num_bytes % (self.item_shape_product * self.element_byte_size)
    
    def set_array_metadata(self, element):

        modified = False

        new_type = element.dtype
        if new_type != self.element_type:
            dtypestr = pickle.dumps(element.dtype)
            self.dtype_str_len.value = len(dtypestr)
            if self.dtype_str_len.value > self.DTYPE_ARRAY_LEN:
                raise RuntimeError('fixed array too small for dtype')
            self.dtype_str_array[0:self.dtype_str_len.value] = dtypestr
            self.element_type = element.dtype
            modified = True

        new_shape = np.asarray(element.shape, dtype='Q')
        if not np.array_equal(new_shape, self.element_shape):
            new_shape = np.asarray(element.shape, dtype='Q')
            self.shape_array_len.value = len(new_shape)
            if self.shape_array_len.value > self.SHAPE_ARRAY_LEN:
                raise RuntimeError('fixed array too small for shape')
            self.shape_array[0:self.shape_array_len.value] = new_shape
            self.element_shape = new_shape
            modified = True

        if modified:    
            self.element_byte_size = self.element_type.itemsize 
            self.item_shape_product = int(np.prod(self.element_shape))
            self.num_items = self.num_bytes // (self.item_shape_product * self.element_byte_size)
            self.dead_bytes = self.num_bytes % (self.item_shape_product * self.element_byte_size)
            with self.lock:
                self.read_cursor.value = 0
                self.write_cursor.value = 0
                self.num_lost_item.value = 0
        
    def get(self, block: bool = True, timeout: Optional[float] = None) -> NDArray:
        '''return buffer to the current read location'''

        if block:
            acquired = self.semaphore.acquire(timeout=timeout)
            if not acquired:
                raise Empty
        else:
            if not self.semaphore.acquire(block=False):
                raise Empty
            
        return self._get_noblock()

    def _get_noblock(self) -> Optional[NDArray]:
        '''return data at the current read location'''
        
        t_start = time.perf_counter_ns() * 1e-6

        with self.lock:

            t_lock_acquired = time.perf_counter_ns() * 1e-6

            if self.empty():
                raise Empty
            
            self.load_array_metadata()

            # get data
            if self.copy:
                element = np.frombuffer(
                    self.data, 
                    dtype = self.element_type, 
                    count = self.item_shape_product,
                    offset = self.read_cursor.value * self.item_shape_product * self.element_byte_size # offset should be in bytes
                ).copy()
            else:
                element = np.frombuffer(
                    self.data, 
                    dtype = self.element_type, 
                    count = self.item_shape_product,
                    offset = self.read_cursor.value * self.item_shape_product * self.element_byte_size # offset should be in bytes
                )
            self.read_cursor.value = (self.read_cursor.value  +  1) % self.num_items

        t_lock_released = time.perf_counter_ns() * 1e-6

        if self.local_logger:
            self.local_logger.info(f'get, {t_start}, {t_lock_acquired}, {t_lock_released}')
        
        return element.reshape(self.element_shape)
    
    def put(self, element: ArrayLike, block: Optional[bool] = True, timeout: Optional[float] = None) -> None:
        '''
        Return data at the current write location.
        block and timeout are there for compatibility with the Queue interface, but 
        are ignored since the ring buffer overflows by design.  
        '''

        t_start = time.perf_counter_ns() * 1e-6

        with self.lock:

            t_lock_acquired = time.perf_counter_ns() * 1e-6

            self.set_array_metadata(element)

            # convert to numpy array
            arr_element = np.asarray(element, dtype = self.element_type)

            buffer = np.frombuffer(
                self.data, 
                dtype = self.element_type, 
                count = self.item_shape_product,
                offset = self.write_cursor.value * self.item_shape_product * self.element_byte_size # offset should be in bytes
            )

            # if the buffer is full, overwrite the next block
            if self.full():
                self.read_cursor.value = (self.read_cursor.value  +  1) % self.num_items
                self.num_lost_item.value += 1

            # write flattened array content to buffer (a copy is made)
            buffer[:] = arr_element.ravel()

            # update write cursor value
            self.write_cursor.value = (self.write_cursor.value  +  1) % self.num_items

        self.semaphore.release()
        t_lock_released = time.perf_counter_ns() * 1e-6

        time.sleep(0.00001) # This might not be necessary in real world, you don't spam put?

        if self.local_logger:
            self.local_logger.info(f'put, {t_start}, {t_lock_acquired}, {t_lock_released}')

    def full(self):
        ''' check if buffer is full '''
        if self.num_items is not None:
            return self.write_cursor.value == ((self.read_cursor.value - 1) % self.num_items)

    def empty(self):
        ''' check if buffer is empty '''
        return self.write_cursor.value == self.read_cursor.value

    def qsize(self):
        ''' Return number of items currently stored in the buffer '''
        with self.lock:
            self.load_array_metadata()
            if self.num_items is not None:
                return (self.write_cursor.value - self.read_cursor.value) % self.num_items
        
    def close(self):
        pass

    def clear(self):
        '''clear the buffer'''
        with self.lock:
            self.write_cursor.value = self.read_cursor.value

    def get_num_items(self) -> Optional[int]:
        with self.lock:
            self.load_array_metadata()
            return self.num_items
    
    def view_data(self):

        if self.element_byte_size is None:
            return np.array([])
        
        num_items = self.write_cursor.value - self.read_cursor.value 

        stored_data = np.frombuffer(                
            self.data, 
            dtype = self.element_type, 
            count = num_items * self.item_shape_product,
            offset = self.read_cursor.value * self.item_shape_product * self.element_byte_size # offset should be in bytes
        ).reshape(np.concatenate(((num_items,) , self.element_shape), dtype=np.int64))

        return stored_data
    
    def __str__(self):

        if self.num_items is None:
            return 'Buffer non-initialized'    
        
        reprstr = (
            f'capacity: {self.num_items - 1}\n' +
            f'dead bytes: {self.dead_bytes}\n' +
            f'data type: {self.element_type}\n' +
            f'data shape: {self.element_shape}\n' +
            f'size: {self.qsize()}\n' +
            f'read cursor position: {self.read_cursor.value}\n' + 
            f'write cursor position: {self.write_cursor.value}\n' +
            f'empty: {self.empty()}\n' +
            f'full: {self.full()}\n' +
            f'lost item: {self.num_lost_item.value}\n' +
            f'buffer: {self.data}\n'
        )

        return reprstr
    
if __name__ == '__main__':

    import numpy as np
    import time
    from multiprocessing import Process

    mrb = ModifiableRingBuffer(
        num_bytes=1024,
        logger = None,
        name = '',
    )
    
    def test(mrb):
        time.sleep(1)
        data = mrb.get()
        print(f'from child process: {(data,)}')
        time.sleep(2)
        data = mrb.get()
        print(f'from child process: {(data,)}')
        time.sleep(1)
        dt = np.dtype([('index', np.uint8, (1,)),('timestamp', np.float32, (1,)),('image', np.uint8, (16,16))])
        mrb.put(np.array([(1, 0.1, np.ones((16,16), dtype = np.uint8))], dtype=dt))
        
    p = Process(target=test, args=(mrb,))
    p.start()
    mrb.put(np.array([0], dtype=np.uint8))
    mrb.put(np.array([1], dtype=np.uint8))
    time.sleep(2)
    mrb.put(np.array([[0, 1, 2],[3, 4, 5]], dtype=np.uint16))
    p.join()

    data = mrb.get()
    print(f'from main process: {(data,)}')