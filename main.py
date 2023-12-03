from keypoint_util import process_keypoints_to_angles, predict_class
from movenet import Movenet
import os
from watchdog import get_new_image_name, delete_image
import joblib


def main():
    image_directory = "captured_image"
    base_directory = os.getcwd()
    model = joblib.load('models/svc_model.sav')
    directory = os.path.join(base_directory, image_directory)
    movenet = Movenet()

    while True:
        image_captured = get_new_image_name(directory, image_directory)
        full_image_dir = os.path.join(directory, image_captured)
        kp = movenet.get_keypoints_with_scores(
            full_image_dir)  # Example with single image
        # kp = keypoints_by_directory(movenet, 'dataset/subset')  # Example with directory
        angles = process_keypoints_to_angles(kp, print_result=True)
        predict_class(angles)
#        if image_captured:
#            delete_image(full_image_dir)


if __name__ == "__main__":
    main()
