from keypoint_util import process_keypoints_to_angles, predict_class
from movenet import Movenet
from misc.video_capture import launch_video_capture


def main():
    movenet = Movenet()
    kp = movenet.get_keypoints_with_scores("dataset/tree/00000003_32.jpg")  # Example with single image
    # kp = keypoints_by_directory(movenet, 'dataset/subset')  # Example with directory
    angles = process_keypoints_to_angles(kp, print_result=True)
    predict_class(angles)


if __name__ == "__main__":
    main()
