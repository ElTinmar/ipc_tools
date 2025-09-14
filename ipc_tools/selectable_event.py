import threading
import time
import select
import socket

class SelectableEvent:
    """A cross-platform selectable event using sockets."""

    def __init__(self):
        self._r, self._w = self._create_socketpair()
        self._r.setblocking(False)
        self._w.setblocking(False)
        self._is_set = False

    @staticmethod
    def _create_socketpair():
        """Create a socket pair cross-platform."""
        if hasattr(socket, "socketpair"):
            return socket.socketpair()
        else:
            # Windows workaround using loopback TCP
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(listener.getsockname())
            server, _ = listener.accept()
            listener.close()
            return server, client

    def fileno(self):
        """Return the file descriptor for select/poll."""
        return self._r.fileno()

    def set(self):
        """Set the event and signal any selectors."""
        if not self._is_set:
            self._is_set = True
            try:
                self._w.send(b'\0')
            except (BlockingIOError, OSError):
                pass  # Already signaled

    def clear(self):
        """Clear the event and drain the socket."""
        self._is_set = False
        try:
            while True:
                data = self._r.recv(1024)
                if not data:
                    break
        except (BlockingIOError, OSError):
            pass

    def is_set(self):
        return self._is_set


WAIT_TIME = 2  


def busy_loop_benchmark():
    print("Running busy loop benchmark...")
    flag = False

    def setter():
        nonlocal flag
        time.sleep(WAIT_TIME)
        flag = True

    threading.Thread(target=setter, daemon=True).start()

    start = time.time()
    while not flag:
        pass  # busy loop
    end = time.time()
    print(f"Busy loop detected flag after {end-start:.4f} seconds")


def selectable_event_benchmark():
    print("Running SelectableEvent benchmark...")
    ev = SelectableEvent()

    def setter():
        time.sleep(WAIT_TIME)
        ev.set()

    threading.Thread(target=setter, daemon=True).start()

    start = time.time()
    rlist, _, _ = select.select([ev], [], [])
    end = time.time()
    print(f"SelectableEvent detected flag after {end-start:.4f} seconds")
    ev.clear()

if __name__ == "__main__":
    busy_loop_benchmark()
    selectable_event_benchmark()
