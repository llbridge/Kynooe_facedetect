from pathlib import Path

# ======================================
# === Base paths ===
# ======================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# === Centralized cache directory ===
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)  # auto-create if missing

# Individual cache files
BLE_CACHE_PATH = CACHE_DIR / "ble_cache.json"
VISION_CACHE_PATH = CACHE_DIR / "vision_cache.json"
STATE_CACHE_PATH = CACHE_DIR / "state.json"

# ======================================
# === Model paths ===
# ======================================
YUNET_ONNX = MODELS_DIR / "yunet.onnx"

# ======================================
# === Stream / capture ===
# ======================================
STREAM_URL = "http://192.168.1.105:80/stream"
CAP_BUFSIZE = 1
REOPEN_SLEEP = 0.2
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX (lazy import in main)

# ======================================
# === Detection parameters ===
# ======================================
INFER_W, INFER_H = 640, 480
SCORE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Schedule
DET_INTERVAL = 5
FPS_PRINT_INTERVAL = 30

# ======================================
# === Face control parameters ===
# ======================================
FACE_CTRL = {
    "default_z": 110.0,
    "z_min": 0.0,
    "z_max": 180.0,
}

# ======================================
# === MediaPipe Hand Tracking ===
# ======================================
MP_HANDS_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.5,
}

# ======================================
# === BLE service / characteristics ===
# ======================================
TARGET_NAME_KEYWORD = "mecharm"
SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
ROLE_UUID = "6d68efe5-04b6-4a85-abc4-c2670b7bf7fd"
JOYSTICK_UUID = "abcd8888-1a2b-3c4d-5e6f-abcdef888888"

DEFAULT_SLAVE_ASSIGNMENTS = [
    "6055F97CB57C=M1-R",
    "50787DF4D808=M2-P",
    # "7CDFA1B20C48=M3-P",
    "50787DF4D85C=M3-P",
    # "6055F97CB4CC=M4-P",
    "34CDB04C3278=M4-P",
    "50787DF4D7EC=M5-G",
]

LINUX_DEFAULT_DEVICE_ADDRESS = "50:78:7d:f4:d8:b2"
LINUX_DEFAULT_ADAPTER_NAME = "hci0"

# === BLE write throttle ===
MIN_WRITE_INTERVAL_SEC = 0.05  # 20 Hz
