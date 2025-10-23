from __future__ import annotations
import mediapipe as mp

def create_hands(static_image_mode: bool, max_num_hands: int, min_detection_confidence: float, min_tracking_confidence: float):
    """Factory for MediaPipe Hands."""
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ), mp_hands, mp.solutions.drawing_utils
