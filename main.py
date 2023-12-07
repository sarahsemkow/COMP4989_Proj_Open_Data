from movenet import Movenet

from video_capture import launch_video_capture
import joblib


def main():
    movenet = Movenet()
    model = joblib.load("models/svc_model.sav")
    launch_video_capture(movenet, model, nano=False)


if __name__ == "__main__":
    main()
