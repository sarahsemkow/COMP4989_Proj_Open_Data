import time

import numpy as np
import tensorflow as tf
import cv2

from constants import KEYPOINT_EDGE_INDS_TO_COLOR
from keypoint_util import predict_class, process_keypoints_to_angles

# No longer needed
def launch_video_capture(movenet, interval=5, threshold=0.1):
    # connects to webcam (can also pass in video here: 'video.mp4'/play around with 0)
    cap = cv2.VideoCapture(0)
    # timer that can takes screenshot every X seconds
    start_time = time.time()
    # image count (if taking screenshots)
    # ss_count = 1

    # loop through every single frame in webcam
    while cap.isOpened():
        # ret = whether a frame was successfully read or not (success status)
        # frame = frame that was read, image represented as arrays [h, w, 3channels]
        ret, frame = cap.read()

        """ reshape image """
        # copy the frame into an image
        img = frame.copy()
        # reshape image, movenet only takes in 192x192 and float32
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        keypoints_with_scores = movenet.get_keypoints_with_scores(input_image)

        """ render """
        movenet.draw_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.2)
        movenet.draw_keypoints(frame, keypoints_with_scores, 0.2)

        # whats captured by webcam
        cv2.imshow("MoveNet Lighting", frame)

        # Check if it's time to take a screenshot
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= interval:
            keypoint_coordinates_within_threshold = keypoints_with_scores[
                keypoints_with_scores[:, 2] > threshold
                ]
            if not keypoint_coordinates_within_threshold.shape[0] < 17:
                # Save the screenshot as an image file/ to replace previous one, just remove {ss_count}
                # cv2.imwrite(f'screenshot{ss_count}.png', frame)
                angles = process_keypoints_to_angles(keypoints_with_scores)
                predict_class(angles)
            else:
                print("Not enough keypoints detected")
            # Increment the screenshot count and reset the timer
            # ss_count += 1
            start_time = time.time()

        # get the key value
        key = cv2.waitKey(10) & 0xFF
        # Break the loop if 'q' is pressed
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
