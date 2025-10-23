# test/servo.py
# Minimal BLE XYZ test using existing BLETransport + wait_for_ble_connection
# Python 3.8+

# test/servo.py
# Minimal BLE XYZ test using existing BLETransport with ready-wait and small resend burst.

import sys, os, time, logging
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to sys.path

from src.ble.transport import BLETransport, wait_for_ble_connection

def wait_for_ready(ble: BLETransport, timeout: float = 3.0, poll: float = 0.05) -> bool:
    """Wait until BLETransport has completed handshake (_ready)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Accessing a protected attr is OK for a test script
        if getattr(ble, "_ready", False):
            return True
        time.sleep(poll)
    return False

def main():
    logging.basicConfig(level=logging.INFO)  # see connect/write logs from transport
    ble = BLETransport()
    ble.start()

    # 1) Wait for link
    if not wait_for_ble_connection(ble, timeout=10.0):
        print("BLE not connected within timeout.")
        ble.stop()
        return

    # 2) Wait for handshake to finish (ready)
    if not wait_for_ready(ble, timeout=3.0):
        print("BLE connected but not ready (handshake not finished).")
        ble.stop()
        return

    # 3) Send XYZ (burst send to be safe)
    payload = {"mode": "rectJoystick", "x": 10, "y": -5, "z": 110, "gripper": 1, "delay": 20}
    for _ in range(3):
        ble.send_json(payload)
        time.sleep(0.1)

    print("Sent:", payload)

    # 4) Give some time to flush then stop
    time.sleep(0.3)
    ble.stop()

if __name__ == "__main__":
    main()
