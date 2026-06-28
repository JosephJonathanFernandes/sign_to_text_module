import cv2
import threading
import time

class CameraThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.running = False
        if self.cap.isOpened():
            # Pre-warm a few frames to avoid the first-frame stall
            for _ in range(3):
                self.cap.read()
            self.running = True
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret:
                # Avoid tight loop if camera fails
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
