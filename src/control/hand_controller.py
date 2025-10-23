from __future__ import annotations

from typing import Optional, Tuple

import cv2

from src.vision.handsdetector import create_hands


__all__ = ["HandController"]


class HandController:
    """Encapsulates MediaPipe hand detection and optional annotation."""

    def __init__(self, *, mp_config: Optional[dict] = None, font=0):
        self.hands, self.mp_hands, self.mp_drawing = create_hands(**(mp_config or {}))
        self.font = cv2.FONT_HERSHEY_SIMPLEX if font == 0 else font
        self.points = [[-2.4, 11.7], [-4.35, 12.65], [-8.61, 10.13], [-12.6, 8.0]]
        self.box_tol = [110.0, 80.0, 50.0]  
        self._current_idx = 0

    def step(
        self,
        frame,
        *,
        annotate: bool = True,
    ) -> Tuple[float, float, Optional[Tuple[int, int]]]:
        """
        Run one detection step, compute follow targets, and optionally annotate the frame.

        Returns:
            x_val, y_val: Follow control targets based on hand proximity.
            hand_point: Optional wrist center (x, y) in pixels.
        """
        x_val, y_val = self.points[self._current_idx]
        hand_point = None
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_height, frame_width = frame.shape[:2]

            wrist = hand_landmarks.landmark[0]
            hx = int(max(0, min(frame_width - 1, wrist.x * frame_width)))
            hy = int(max(0, min(frame_height - 1, wrist.y * frame_height)))
            hand_point = (hx, hy)

            xs = [lm.x * frame_width for lm in hand_landmarks.landmark]
            if xs:
                min_x = min(xs)
                max_x = max(xs)
                bbox_width = max_x - min_x
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
                cv2.circle(frame, hand_point, 8, (0, 255, 0), -1)
                cv2.putText(frame, "hand detection", (hx + 10, max(hy - 10, 20)), self.font, 0.6, (0, 255, 0), 2)
        else:
            x_val, y_val = self.points[self._current_idx]
            print(f"No hand detected, holding index {self._current_idx} -> ({x_val}, {y_val}).")

        return float(x_val), float(y_val), hand_point

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
