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
    model = joblib.load('models/svc_model.sav')

    # FOR JETBOT
    #     image_directory = "captured_image"
    #     base_directory = os.getcwd()
    #     directory = os.path.join(base_directory, image_directory)
    #
    #     while True:
    #         image_captured = get_new_image_name(directory, image_directory)
    #         full_image_dir = os.path.join(directory, image_captured)
    #         kp = movenet.get_keypoints_with_scores(
    #             full_image_dir)  # Example with single image
    #         # kp = keypoints_by_directory(movenet, 'dataset/subset')  # Example with directory
    #         angles = process_keypoints_to_angles(kp, print_result=True)
    #         predict_class(model, angles)
    # #        if image_captured:
    # #            delete_image(full_image_dir)

    # FOR DEVELOPING ON COMPUTER

    launch_video_capture(movenet, model, nano=False)
    # # process image
    # image = movenet.process_image('goddess-good.jpg')
    # # pass processed image to keypoints
    # kp = movenet.get_keypoints_with_scores(image)  # Example with single image
    # # kp = keypoints_by_directory(movenet, 'dataset/subset')  # Example with directory
    # angles = process_keypoints_to_angles(kp, print_result=True)
    # model_probabilities = predict_class(model, angles) # gives a dataframe with all probabilities
    # predicted_label = model_probabilities['True Label'].iloc[0] # gets the predicted model label
    # # feedback
    # feedback, feedback_reasons = evaluatePose(predicted_label, angles, kp)
    # print(feedback)
    # print(feedback_reasons)

if __name__ == "__main__":
    main()
