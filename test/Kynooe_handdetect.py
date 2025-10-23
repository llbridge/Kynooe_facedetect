import cv2
from pathlib import Path
import mediapipe as mp

# ========= Config =========
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "handimage.jpg"    # put your image next to this script

# ========= MediaPipe setup =========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def main():
    image = cv2.imread(str(IMAGE_PATH))
    h, w = image.shape[:2]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    handLms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                wrist = handLms.landmark[0]
                x, y = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
        else:
            print("No hand detected.")

    cv2.imshow("Hand Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
