from __future__ import annotations

from typing import Optional, Tuple

import cv2

import config
from src.vision.handsdetector import create_hands


__all__ = ["HandController"]


class HandController:
    """Encapsulates MediaPipe hand detection and optional annotation."""

    def __init__(self, *, mp_config: Optional[dict] = None, font=0, ctrl_cfg: Optional[dict] = None):
        self.hands, self.mp_hands, self.mp_drawing = create_hands(**(mp_config or {}))
        self.font = cv2.FONT_HERSHEY_SIMPLEX if font == 0 else font
        self.points = [[-2.4, 11.7], [-4.35, 12.65], [-8.61, 10.13], [-12.6, 8.0]]
        self.box_tol = [110.0, 80.0, 50.0]  
        self._current_idx = 0
        base_cfg = ctrl_cfg or config.HAND_CTRL
        self.ctrl_cfg = dict(base_cfg)
        self.default_z = float(self.ctrl_cfg["default_z"])
        self.z_min = float(self.ctrl_cfg["z_min"])
        self.z_max = float(self.ctrl_cfg["z_max"])
        self.tolerance_px = float(self.ctrl_cfg["tolerance_px"])
        self.min_step = float(self.ctrl_cfg["min_step"])
        self.max_step = float(self.ctrl_cfg["max_step"])
        self.z_gain = float(self.ctrl_cfg["z_gain"])

    def step(
        self,
        frame,
        *,
        annotate: bool = True,
    ) -> Tuple[float, float, float, Optional[Tuple[int, int]]]:
        """
        Run one detection step, compute follow targets, and optionally annotate the frame.

        Returns:
            x_val, y_val: Follow control targets based on hand proximity.
            z_step: Incremental z control (float) based on horizontal offset.
            hand_point: Optional bounding-box center (x, y) in pixels.
        """
        x_val, y_val = self.points[self._current_idx]
        z_step = 0.0
        hand_point = None
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_height, frame_width = frame.shape[:2]

            xs = [lm.x * frame_width for lm in hand_landmarks.landmark]
            ys = [lm.y * frame_height for lm in hand_landmarks.landmark]
            if xs and ys:
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                bbox_width = max_x - min_x
                bbox_center_x = (min_x + max_x) / 2.0
                bbox_center_y = (min_y + max_y) / 2.0
                hand_point = (int(bbox_center_x), int(bbox_center_y))

                center_x = frame_width / 2.0
                z_diff = bbox_center_x - center_x
                abs_diff = abs(z_diff)
                if abs_diff <= self.tolerance_px:
                    print(
                        f"Hand centered: offset {abs_diff:.2f}px "
                        f"(tolerance {self.tolerance_px:.2f}px)"
                    )
                    z_step = 0.0
                else:
                    overshoot = abs_diff - self.tolerance_px
                    raw_step = overshoot * self.z_gain
                    step_mag = float(min(self.max_step, max(self.min_step, raw_step)))
                    z_step = -step_mag if z_diff > 0 else step_mag
                    direction = "right" if z_diff > 0 else "left"
                    print(
                        f"BBox center offset {abs_diff:.2f}px (tolerance {self.tolerance_px:.2f}px) -> "
                        f"overshoot {overshoot:.2f}px | direction {direction} | z step {z_step:.2f}"
                    )

                box_tol_txt = " >= ".join(f"{t:.1f}" for t in self.box_tol)
                print(
                    f"x[{min_x:.2f},{max_x:.2f}]px width:{bbox_width:.2f}px | "
                    f"width-thresholds:{box_tol_txt}"
                )
                selection_idx = len(self.points) - 1
                for idx, threshold in enumerate(self.box_tol):
                    if bbox_width >= threshold:
                        selection_idx = idx
                        break

                self._current_idx = selection_idx
                x_val, y_val = self.points[self._current_idx]
                print(f"Selected follow target index {self._current_idx} -> ({x_val}, {y_val})")

            if annotate:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if hand_point:
                    top_left = (int(min_x), int(min_y))
                    bottom_right = (int(max_x), int(max_y))
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
                    cv2.circle(frame, hand_point, 6, (0, 255, 0), -1)
                    label_pos = (hand_point[0] + 10, max(hand_point[1] - 10, 20))
                    cv2.putText(frame, "hand bbox", label_pos, self.font, 0.6, (0, 255, 0), 2)
        else:
            x_val, y_val = self.points[self._current_idx]
            print(f"No hand detected, holding index {self._current_idx} -> ({x_val}, {y_val}).")

        return float(x_val), float(y_val), float(z_step), hand_point

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
