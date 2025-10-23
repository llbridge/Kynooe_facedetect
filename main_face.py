from __future__ import annotations

import queue
import threading
import time

import cv2

from config import FONT, FPS_PRINT_INTERVAL, STREAM_URL, YUNET_ONNX
from src.ble.transport import (
    ADAPTER_NAME,
    BLETransport,
    FIXED_DEVICE_ADDRESS,
    wait_for_ble_connection,
)
from src.control.face_controller import FaceController
from src.utils.env import parse_timeout_env
from src.vision.capture import CaptureWorker
from src.vision.facedetector import FaceDetectorTracker, assert_model_ok


FACE_FIXED_X = -8.61
FACE_FIXED_Y = 10.13
BLE_CONNECT_TIMEOUT_ENV = "MECHARM_BLE_CONNECT_TIMEOUT"
WINDOW_TITLE = "yunet face control"
FPS_LABEL_POS = (10, 30)


def start_capture(stream_url: str) -> tuple[queue.Queue, threading.Event, CaptureWorker]:
    """Start capture worker and return queue, stop-event, thread."""
    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = CaptureWorker(stream_url, frame_queue, stop_event)
    worker.start()
    return frame_queue, stop_event, worker


def wait_for_first_frame(frame_queue: queue.Queue, *, timeout: float = 2.0) -> None:
    """Block until the first frame arrives."""
    frame = None
    while frame is None:
        try:
            frame = frame_queue.get(timeout=timeout)
        except queue.Empty:
            print("waiting for stream...")


def build_payload(x_val: float, y_val: float, z_val: float) -> dict:
    """Construct BLE joystick payload."""
    return {
        "mode": "rectJoystick",
        "x": float(x_val),
        "y": float(y_val),
        "z": float(z_val),
        "gripper": 1,
        "delay": 20,
    }


def update_fps(counter: int, prev_ts: float, interval: int) -> tuple[int, float, float]:
    """Update FPS counters and return (counter, prev_ts, fps)."""
    counter += 1
    if counter < interval:
        return counter, prev_ts, 0.0

    now = time.time()
    elapsed = max(1e-6, now - prev_ts)
    fps = counter / elapsed
    return 0, now, fps


def main() -> None:
    """Entry point for face tracking control."""
    assert_model_ok(YUNET_ONNX)

    frame_queue, stop_event, cap_thread = start_capture(STREAM_URL)
    wait_for_first_frame(frame_queue)

    face_tracker = FaceDetectorTracker(YUNET_ONNX)
    face_ctrl = FaceController(face_tracker, font=FONT)
    print("yunet + face control initialized. press 'q' to quit.")

    ble_transport = BLETransport()
    ble_transport.start()
    print(f"ble config: address={FIXED_DEVICE_ADDRESS or 'scan-by-name/uuid'}, adapter={ADAPTER_NAME or 'system-default'}")

    prev_time = time.time()
    fps = 0.0
    frame_counter = 0

    try:
        print("waiting for bluetooth connection...")
        timeout_override = parse_timeout_env(BLE_CONNECT_TIMEOUT_ENV)
        if not wait_for_ble_connection(
            ble_transport,
            timeout=timeout_override if timeout_override is not None else 35.0,
        ):
            print("bluetooth not connected within timeout; exiting.")
            return

        face_z = 110

        while True:
            try:
                frame = frame_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            x_val, y_val, z_val, face_center = face_ctrl.step(frame)
            face_z = face_z + z_val

            if ble_transport.connected():
                payload = build_payload(x_val, y_val, face_z)
                if face_center:
                    print(f"[SEND] x={FACE_FIXED_X:.2f}, y={FACE_FIXED_Y:.2f}, z={z_val:.2f}")
                else:
                    print(f"[SEND] x={FACE_FIXED_X:.2f}, y={FACE_FIXED_Y:.2f}, z={z_val:.2f} (no face)")
                ble_transport.send_json(payload)

            frame_counter, prev_time, fps = update_fps(frame_counter, prev_time, FPS_PRINT_INTERVAL)

            cv2.putText(frame, f"fps: {fps:.2f}", FPS_LABEL_POS, face_ctrl.font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("exiting...")
                break

    finally:
        stop_event.set()
        cap_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        ble_transport.stop()


if __name__ == "__main__":
    main()
