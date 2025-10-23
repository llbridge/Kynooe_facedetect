# Minimal HaGRID YOLO gesture detection from HTTP stream (YuNet-like)
# Notes:
# - English-only comments.
# - Replace MODEL_PATH with your HaGRID YOLOv8/YOLOv10 weights path.
# - STREAM_URL should point to your ESP32-CAM or MJPEG HTTP stream.

import cv2
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent         
IMAGE_PATH = BASE_DIR / "faceimage.jpg"            
YUNET_ONNX = BASE_DIR.parent / "models" / "yunet.onnx"  

INFER_W, INFER_H = 320, 240
SCORE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
TOP_K = 5000
FONT = cv2.FONT_HERSHEY_SIMPLEX

def main():
    image = cv2.imread(str(IMAGE_PATH))
    h, w = image.shape[:2]

    det = cv2.FaceDetectorYN.create(
        model=str(YUNET_ONNX),
        config="",
        input_size=(INFER_W, INFER_H),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
    )

    resized = cv2.resize(image, (INFER_W, INFER_H))
    _, faces = det.detect(resized)

    if faces is None or len(faces) == 0:
        return
    sx, sy = w / INFER_W, h / INFER_H
    faces[:, [0, 2, 5, 7, 9, 11, 13]] *= sx
    faces[:, [1, 3, 6, 8, 10, 12, 14]] *= sy
    for i, f in enumerate(faces):
        x, y, fw, fh, score = f[:5]
        cv2.rectangle(image, (int(x), int(y)), (int(x + fw), int(y + fh)), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (int(x), int(y) - 5), FONT, 0.6, (0, 255, 0), 2)
    cv2.imshow("Face Detection", image)
    cv2.imwrite(str(BASE_DIR / "face_detected.jpg"), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
