from __future__ import annotations

import queue
import threading
import time

import cv2

from config import (
    STREAM_URL,
    FPS_PRINT_INTERVAL,
    MP_HANDS_CONFIG,
    FONT,
)
from src.ble.transport import (
    ADAPTER_NAME,
    BLETransport,
    FIXED_DEVICE_ADDRESS,
    wait_for_ble_connection,
)
from src.control.hand_controller import HandController
from src.utils.env import parse_timeout_env
from src.vision.capture import CaptureWorker


def main() -> None:
    # Start camera capture thread
    q: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    cap_thread = CaptureWorker(STREAM_URL, q, stop_event)
    cap_thread.start()

    # Wait for first frame
    frame = None
    while frame is None:
        try:
            frame = q.get(timeout=2.0)
        except queue.Empty:
            print("waiting for stream...")

    # Initialize controller
    hand_ctrl = HandController(mp_config=MP_HANDS_CONFIG, font=FONT)
    print("mediapipe hand detection initialized. press 'q' to quit.")

    # Start BLE transport
    ble_transport = BLETransport()
    ble_transport.start()
    print(f"ble config: address={FIXED_DEVICE_ADDRESS or 'scan-by-name/uuid'}, adapter={ADAPTER_NAME or 'system-default'}")

    prev_time = time.time()
    fps = 0.0
    frame_counter = 0

    try:
        print("waiting for bluetooth connection...")
        timeout_override = parse_timeout_env("MECHARM_BLE_CONNECT_TIMEOUT")
        if not wait_for_ble_connection(ble_transport, timeout=timeout_override if timeout_override is not None else 35.0):
            print("bluetooth not connected within timeout; exiting.")
            return

        z_val = hand_ctrl.default_z

        while True:
            try:
                frame = q.get(timeout=2.0)
            except queue.Empty:
                continue

            frame = cv2.flip(frame, 1)

            x_val, y_val, z_step, hand_point = hand_ctrl.step(frame)
            if z_step != 0.0:
                z_val += z_step
            else:
                if z_val < hand_ctrl.default_z:
                    z_val = min(hand_ctrl.default_z, z_val + 1.0)
                elif z_val > hand_ctrl.default_z:
                    z_val = max(hand_ctrl.default_z, z_val - 1.0)

            z_val = max(hand_ctrl.z_min, min(hand_ctrl.z_max, z_val))

            if ble_transport.connected():
                payload = {
                    "mode": "rectJoystick",
                    "x": float(x_val),
                    "y": float(y_val),
                    "z": float(z_val),
                    "gripper": 1,
                    "delay": 20,
                }
                if hand_point:
                    print(f"[SEND] x={x_val:.2f}, y={y_val:.2f}, z={z_val:.2f}")
                else:
                    print(f"[SEND] x={x_val:.2f}, y={y_val:.2f}, z={z_val:.2f} (no hand)")
                ble_transport.send_json(payload)

            frame_counter += 1
            if frame_counter >= FPS_PRINT_INTERVAL:
                now = time.time()
                fps = frame_counter / max(1e-6, (now - prev_time))
                prev_time = now
                frame_counter = 0

            display_text = f"fps: {fps:.2f} | x:{x_val:.2f} y:{y_val:.2f} z:{z_val:.2f}"
            cv2.putText(frame, display_text, (10, 30), hand_ctrl.font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("mediapipe hand detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("exiting...")
                break

    finally:
        hand_ctrl.close()
        stop_event.set()
        cap_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        ble_transport.stop()


if __name__ == "__main__":
    main()
