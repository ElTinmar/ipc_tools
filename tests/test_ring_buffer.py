import unittest
import numpy as np
from ipc_tools import RingBuffer, ModifiableRingBuffer
from multiprocessing import Process, Manager
from queue import Empty, Full
import time

SZ = (1024, 1024)
TS = 10.0
ARRAY = np.random.uniform(low=-1, high=1, size=SZ).astype(np.float32)

class Tests(unittest.TestCase):

    def test_array(self):
        buf = RingBuffer(
            num_items=100,
            item_shape=SZ,
            data_type=np.float32
        )
        buf.put(ARRAY)
        res = buf.get()
        self.assertTrue(np.allclose(res, ARRAY))

    def test_structured_array_0(self):
        dt = np.dtype([
            ('timestamp', np.float64, (1,)), 
            ('image', np.float32, SZ)
        ])
        x = np.array([(TS, ARRAY)], dtype=dt)
        buf = RingBuffer(
            num_items=100,
            item_shape=(1,),
            data_type=dt
        )
        buf.put(x)
        res = buf.get()
        self.assertEqual(TS, res['timestamp'])
        self.assertTrue(np.allclose(res['image'], x['image']))

    
    def test_structured_array_1(self):
        dt = np.dtype([
            ('timestamp', np.float64, (1,)), 
            ('image', np.float32, SZ)
        ])

        Array_0 = np.random.uniform(low=-1, high=1, size=SZ).astype(np.float32)
        Array_1 = np.random.uniform(low=-1, high=1, size=SZ).astype(np.float32)

        x0 = np.array([(TS, Array_0)], dtype=dt) 
        x1 = np.array((TS, Array_1), dtype=dt) # no need for [] if only one element
        buf = RingBuffer(
            num_items=100,
            item_shape=(1,),
            data_type=dt
        )
        buf.put(x0)
        buf.put(x1)
        res0 = buf.get()
        res1 = buf.get()

        self.assertTrue(np.allclose(res0['image'], Array_0))
        self.assertTrue(np.allclose(res1['image'], Array_1))

class TestModifiableRingBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer_size = 1024 * 64  # 64 KB
        self.ring = ModifiableRingBuffer(num_bytes=self.buffer_size, t_refresh=1e-4, copy=True)

    def test_put_and_get_single(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        self.ring.put(data)
        result = self.ring.get()
        np.testing.assert_array_equal(result, data)

    def test_buffer_overwrite(self):
        data = np.array([1], dtype=np.float32)
        self.ring.put(data)
        capacity = self.ring.get_num_items()

        for i in range(capacity + 5):  # overflow
            self.ring.put(np.array([i], dtype=np.float32))

        # Only (capacity - 1) should be stored
        self.assertEqual(self.ring.qsize(), capacity - 1)
        self.assertGreater(self.ring.num_lost_item.value, 0)

        items = [self.ring.get()[0] for _ in range(self.ring.qsize())]
        self.assertEqual(items[-1], capacity + 4)  # last inserted item should be last retrieved

    def test_put_large_item(self):
        large_item = np.ones(self.buffer_size + 1, dtype=np.float32)  
        with self.assertRaises(Full): 
            self.ring.put(large_item)  

    def test_put_large_item_qsize(self):
        large_item = np.ones(self.buffer_size + 1, dtype=np.float32)  
        self.ring.put(large_item)  
        self.assertEqual(self.ring.qsize(), 0)

    def test_different_shapes(self):
        a = np.ones((4,), dtype=np.int32)
        self.ring.put(a)
        self.assertTrue(np.array_equal(self.ring.get(), a))

        b = np.ones((2, 2), dtype=np.int32)
        self.ring.put(b)
        result = self.ring.get()
        self.assertEqual(result.shape, (2, 2))

    def test_different_dtypes(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        self.ring.put(a)
        _ = self.ring.get()

        b = np.array([1, 2, 3], dtype=np.int16)
        self.ring.put(b)
        result = self.ring.get()
        self.assertEqual(result.dtype, np.int16)

    def test_get_empty_nonblocking(self):
        with self.assertRaises(Empty):
            self.ring.get(block=False)

    def test_get_timeout(self):
        start = time.monotonic()
        with self.assertRaises(Empty):
            self.ring.get(timeout=0.1)
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.1)

    def test_qsize_tracking(self):
        a = np.ones((4,), dtype=np.float32)
        self.ring.put(a)
        self.ring.put(a)
        self.assertEqual(self.ring.qsize(), 2)
        _ = self.ring.get()
        self.assertEqual(self.ring.qsize(), 1)

    def test_qsize_empty(self):
        self.assertIsNone(self.ring.qsize())

    def test_clear(self):
        a = np.ones((4,), dtype=np.float32)
        self.ring.put(a)
        self.ring.put(a)
        self.ring.clear()
        self.assertEqual(self.ring.qsize(), 0)
        with self.assertRaises(Empty):
            self.ring.get(block=False)

    def test_view_data(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        self.ring.put(a)
        self.ring.put(a)
        view = self.ring.view_data()
        self.assertEqual(view.shape[1:], a.shape)
        self.assertGreaterEqual(view.shape[0], 1)  # may vary depending on overwrites


class TestModifiableRingBufferWithMultiprocessing(unittest.TestCase):

    def setUp(self):
        self.buffer_size = 1024 * 64  # 64 KB
        self.ring = ModifiableRingBuffer(num_bytes=self.buffer_size, t_refresh=1e-4, copy=True)

    def test_multiprocess_put_and_get(self):
        """Test the buffer with multiple producers and consumers."""
        
        def producer(ring, num_items):
            for i in range(num_items):
                item = np.array([i], dtype=np.float32)
                ring.put(item)
        
        def consumer(ring, num_items, result_list):
            for _ in range(num_items):
                item = ring.get(block=True)
                result_list.append(item[0])  # Extract value from array
        
        num_items = 10
        manager = Manager()
        result_list = manager.list()  # Shared list to collect consumer results

        # Create producer and consumer processes
        producer_processes = [Process(target=producer, args=(self.ring, num_items)) for _ in range(3)]
        consumer_processes = [Process(target=consumer, args=(self.ring, num_items, result_list)) for _ in range(3)]
        print(consumer_processes)

        # Start all processes
        for p in producer_processes + consumer_processes:
            p.start()

        # Wait for all processes to finish
        for p in producer_processes + consumer_processes:
            p.join()

        # Assert the results are as expected
        self.assertEqual(len(result_list), num_items * len(consumer_processes))
        result_set = set(result_list)
        self.assertEqual(len(result_set), num_items)  # Ensure all unique items were consumed
        
    def test_buffer_overwrite_with_multiprocessing(self):
        """Test that the buffer overwrites old data when full in a multiprocess scenario."""
        
        def producer(ring, num_items):
            for i in range(num_items):
                item = np.array([i], dtype=np.float32)
                ring.put(item)

        # Start by writing more data than the buffer can hold
        data = np.array([1], dtype=np.float32)
        self.ring.put(data)
        num_items = self.ring.get_num_items() + 10
        producer_processes = [Process(target=producer, args=(self.ring, num_items)) for i in range(2)]

        # Start producer processes
        for p in producer_processes:
            p.start()
        
        # Wait for all producers to finish
        for p in producer_processes:
            p.join()

        # Now check the buffer state
        self.assertGreater(self.ring.num_lost_item.value, 0)
        self.assertEqual(self.ring.qsize(), self.ring.get_num_items() - 1)
        
    def test_get_timeout_with_multiprocessing(self):
        """Test that get raises an Empty exception when the buffer is empty after a timeout in a multiprocess scenario."""

        def consumer(ring, result_list):
            with self.assertRaises(Empty):
                ring.get(timeout=0.1)
            result_list.append('done')  # Indicate that the consumer tried to get

        manager = Manager()
        result_list = manager.list()

        # Start a consumer process and ensure it tries to get from an empty buffer
        consumer_process = Process(target=consumer, args=(self.ring, result_list))
        consumer_process.start()
        consumer_process.join()

        self.assertIn('done', result_list)  # Ensure the consumer has completed the test

    def test_multiprocess_buffer_clear(self):
        """Test that clear works correctly with multiple processes."""
        
        def producer(ring, num_items):
            for i in range(num_items):
                item = np.array([i], dtype=np.float32)
                ring.put(item)
        
        def consumer(ring, num_items, result_list):
            for i in range(num_items):
                item = ring.get(block=True)
                result_list.append(item[0])
        
        num_put = 5
        num_get = 3
        manager = Manager()
        result_list = manager.list()

        # Start producer processes
        producer_processes = [Process(target=producer, args=(self.ring, num_put))]
        consumer_processes = [Process(target=consumer, args=(self.ring, num_get, result_list))]
        for p in producer_processes + consumer_processes:
            p.start()
        for p in producer_processes + consumer_processes:
            p.join()

        # Ensure that consumers have successfully retrieved data before clearing
        self.assertEqual(len(result_list), num_get)

        # Clear the buffer and verify
        self.ring.clear()

        # Verify that the buffer is empty after clearing
        with self.assertRaises(Empty):
            self.ring.get(block=True)

if __name__ == '__main__':
    unittest.main()