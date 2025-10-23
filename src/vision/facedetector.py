from __future__ import annotations
from typing import Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import cv2
import config 


def assert_model_ok(path: Path) -> None:
    """Ensure ONNX model exists and is plausible size."""
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    if path.stat().st_size < 200_000:
        raise ValueError(f"model file too small ({path.stat().st_size} bytes). re-download the onnx.")


def _create_tracker():
    """Create a single-object tracker. Prefer CSRT, fallback to KCF."""
    tracker = None
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        tracker = cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        tracker = cv2.legacy.TrackerKCF_create()
    else:
        factory = getattr(cv2, "TrackerCSRT_create", None) or getattr(cv2, "TrackerKCF_create", None)
        tracker = factory() if factory else None
    if tracker is None:
        raise RuntimeError("no tracker implementation available (CSRT/KCF).")
    return tracker


class FaceDetectorTracker:
    """Run YuNet every N frames and track in-between."""
    def __init__(self, model_path: Path):
        self.det = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(config.INFER_W, config.INFER_H),
            score_threshold=config.SCORE_THRESHOLD,
            nms_threshold=config.NMS_THRESHOLD,
            top_k=config.TOP_K,
        )
        self.trackers: List[Any] = []
        self.tracks_scores: List[float] = []
        self.frame_index = 0

    @staticmethod
    def _scale_faces_back(faces: np.ndarray, sx: float, sy: float) -> np.ndarray:
        f = faces.copy()
        f[:, 0] *= sx
        f[:, 1] *= sy
        f[:, 2] *= sx
        f[:, 3] *= sy
        for i in range(5):
            f[:, 5 + 2 * i] *= sx
            f[:, 6 + 2 * i] *= sy
        return f

    def detect_faces(self, frame: np.ndarray) -> Optional[np.ndarray]:
        small = cv2.resize(frame, (config.INFER_W, config.INFER_H), interpolation=cv2.INTER_LINEAR)
        _, faces = self.det.detect(small)
        if faces is not None and len(faces) > 0:
            h, w = frame.shape[:2]
            faces = self._scale_faces_back(faces, w / config.INFER_W, h / config.INFER_H)
        return faces

    def reset_trackers(self, frame: np.ndarray, faces: np.ndarray):
        self.trackers.clear()
        self.tracks_scores.clear()
        for f in faces:
            x, y, w, h = f[:4].astype(int)
            if w <= 0 or h <= 0:
                continue
            trk = _create_tracker()
            trk.init(frame, (x, y, w, h))
            self.trackers.append(trk)
            self.tracks_scores.append(float(f[4]))

    def update_trackers(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        bboxes, alive_trackers, alive_scores = [], [], []
        for trk, sc in zip(self.trackers, self.tracks_scores):
            ok, box = trk.update(frame)
            if ok:
                x, y, w, h = box
                bboxes.append((int(x), int(y), int(w), int(h)))
                alive_trackers.append(trk)
                alive_scores.append(sc)
        self.trackers, self.tracks_scores = alive_trackers, alive_scores
        return bboxes

    def step(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float], Optional[np.ndarray]]:
        use_detector = (self.frame_index % config.DET_INTERVAL == 0) or (len(self.trackers) == 0)
        self.frame_index += 1

        if use_detector:
            faces = self.detect_faces(frame)
            if faces is not None and len(faces) > 0:
                self.reset_trackers(frame, faces)
                det_boxes = [tuple(f[:4].astype(int)) for f in faces]
                det_scores = [float(f[4]) for f in faces]
                return det_boxes, det_scores, faces

        boxes = self.update_trackers(frame)
        return boxes, self.tracks_scores[:], None
