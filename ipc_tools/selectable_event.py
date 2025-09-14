import multiprocessing
import time
import select
import socket

class SelectableEvent:
    """A cross-platform selectable event using sockets."""

    def __init__(self):
        self._r, self._w = socket.socketpair() # requires python >3.5 on Windows
        self._r.setblocking(False)
        self._w.setblocking(False)
        self._is_set = multiprocessing.Value('b', False)

    def fileno(self):
        """Return the file descriptor for select/poll."""
        return self._r.fileno()

    def set(self):
        """Set the event and signal any selectors/processes."""
        with self._is_set.get_lock():
            if not self._is_set.value:
                self._is_set.value = True
                try:
                    self._w.send(b'\0')
                except (BlockingIOError, OSError):
                    pass  # Already signaled

    def clear(self):
        """Clear the event and drain the socket."""
        with self._is_set.get_lock():
            self._is_set.value = False
            try:
                while True:
                    data = self._r.recv(1024)
                    if not data:
                        break
            except (BlockingIOError, OSError):
                pass

    def is_set(self):
        with self._is_set.get_lock():
            return self._is_set.value
    
    def __del__(self):
        try:
            self._r.close()
        except Exception:
            pass
        try:
            self._w.close()
        except Exception:
            pass
        try:
            del self._is_set
        except Exception:
            pass

def worker(ev, delay):
    time.sleep(delay)
    ev.set()
    print(f"Worker set event after {delay} seconds")

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    ev1 = SelectableEvent()
    ev2 = SelectableEvent()

    # Pass the _w socket to child processes for signaling
    multiprocessing.Process(target=worker, args=(ev1, 1)).start()
    multiprocessing.Process(target=worker, args=(ev2, 2)).start()

    # Wait on multiple events
    while True:
        rlist, _, _ = select.select([ev1, ev2], [], [], 3)
        if not rlist:
            print("Timeout, no events")
            break
        for ev in rlist:
            print(f"{ev} ready")
            ev.clear()