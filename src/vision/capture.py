from __future__ import annotations
import time
import threading
import queue
from typing import Optional

import cv2
import config  


def open_capture(url: str) -> cv2.VideoCapture:
    """Open camera/stream and set a small buffer for low latency."""
    cap = cv2.VideoCapture(url if url.startswith(("rtsp://", "http://")) else int(url))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAP_BUFSIZE)
    return cap


class CaptureWorker(threading.Thread):
    """Background frame reader with tiny queue for low latency."""
    def __init__(self, url: str, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.url = url
        self.q = out_queue
        self.stop_event = stop_event

    def _reopen(self, cap: Optional[cv2.VideoCapture]) -> cv2.VideoCapture:
        if cap:
            cap.release()
        time.sleep(config.REOPEN_SLEEP)
        return open_capture(self.url)

    def run(self):
        cap = open_capture(self.url)
        while not self.stop_event.is_set():
            if not cap.isOpened():
                cap = self._reopen(cap)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap = self._reopen(cap)
                continue
            try:
                if self.q.full():
                    _ = self.q.get_nowait()  # drop oldest
                self.q.put_nowait(frame)
            except queue.Full:
                pass
        cap.release()
