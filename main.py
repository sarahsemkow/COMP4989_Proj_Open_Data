import sklearn
import tensorflow as tf
from keypoint_util import process_keypoints_to_angles, predict_class
from movenet import Movenet
from feedback import evaluatePose
import os

from video_capture import launch_video_capture
from watchdog import get_new_image_name, delete_image
import joblib


def main():
    movenet = Movenet()
    model = joblib.load("models/svc_model.sav")

    launch_video_capture(movenet, model, nano=False)



if __name__ == "__main__":
    main()
