from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import config


__all__ = ["FaceController", "calculate_z"]


def calculate_z(
    x_position: int,
    frame_width: int,
    *,
    cfg=None,
    last_z: Optional[float] = None,
    log: bool = False,
) -> float:
    """Map face x offset to z angle within controller bounds."""
    params = cfg or config.FACE_CTRL
    default_z = float(params["default_z"])
    z_min = float(params["z_min"])
    z_max = float(params["z_max"])

    if frame_width <= 0:
        return default_z

    center_x = frame_width / 2.0
    offset_px = float(x_position) - center_x

    if center_x <= 0:
        return default_z

    normalized = np.clip(offset_px / center_x, -1.0, 1.0)
    if normalized >= 0:
        # move right -> decrease z
        delta = normalized * (default_z - z_min)
        z = default_z - delta
    else:
        # move left -> increase z
        delta = abs(normalized) * (z_max - default_z)
        z = default_z + delta

    z = float(np.clip(z, z_min, z_max))
    if log:
        print(f"x: {x_position}, center_x: {center_x}, z: {z}")
    return z


class FaceController:
    """Encapsulates face tracking and z-angle command generation."""

    def __init__(self, tracker, *, font=0, ctrl_cfg=None, default_z: Optional[float] = None):
        self.tracker = tracker
        self.font = cv2.FONT_HERSHEY_SIMPLEX if font == 0 else font

        base_cfg = ctrl_cfg or config.FACE_CTRL
        self.ctrl_cfg = dict(base_cfg)
        if default_z is not None:
            self.ctrl_cfg["default_z"] = float(default_z)

        self.default_z = float(self.ctrl_cfg["default_z"])
        self.z_min = float(self.ctrl_cfg["z_min"])
        self.z_max = float(self.ctrl_cfg["z_max"])

        self._last_z: float = self.default_z
        self.points = [[-4.35, 12.65], [-8.61, 10.13]]

    def step(
        self,
        frame,
        *,
        annotate: bool = True,
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Run one control step.

        Returns:
            z_val: Roll command for the gripper.
            face_center: Optional face center (x, y) in pixels.
        """
        boxes, _, _ = self.tracker.step(frame)
        print("Detected boxes:", boxes)
        face_center: Optional[Tuple[int, int]] = None
        if boxes:
            x, y, w, h = boxes[0]
            face_center = (x + w // 2, y + h // 2)
            print("Face center:", face_center)
            if annotate:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, face_center, 6, (0, 200, 255), -1)
                cv2.putText(frame, "face detection", (x, max(y - 10, 20)), self.font, 0.7, (0, 255, 0), 2)
        else:
            print("No face detected.")
            x, y, w, h = 0, 0, 0, 0

        box_tol = 100
        if w > box_tol:
            print("Face too close, setting z to default.")
            x_val = self.points[0][0]
            y_val = self.points[0][1]
        else:
            print("Face at normal distance, setting z to adjusted value.")
            x_val = self.points[1][0]
            y_val = self.points[1][1]
        
        frame_width = frame.shape[1]

        z_diff = face_center[0] - (frame_width / 2) if face_center else 0
        print("Z difference:", z_diff)
        tolerance = 30  # pixels
        if abs(z_diff) <= tolerance:
            print("Face is centered.")
            z_val = 0
        elif z_diff > 0:
            print("Face is to the right of center.")
            z_val = 1
        else:
            print("Face is to the left of center.")
            z_val = -1
        return x_val, y_val, z_val, face_center
