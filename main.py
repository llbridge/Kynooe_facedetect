# main.py
from main_face import main as main_face
from main_hand import main as main_hand

__all__ = ["main_face", "main_hand"]


def main() -> None:
    raise SystemExit("Use main_face.main() or main_hand.main() instead of the combined entrypoint.")


if __name__ == "__main__":
    main()
