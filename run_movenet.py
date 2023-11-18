# Import TF and TF Hub libraries.
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def movenet(image_path):
    # Load the input image.
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 192, 192)

    # Initialize the TFLite interpreter
    model_path = '3.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    # Output is a [1, 1, 17, 3] numpy array.
    # Keypoint coordinates are in the range [0.0, 1.0] normalized by the input image size.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores


def plot_keypoints_and_save(keypoints_with_scores, threshold, path=None):
    reshaped = keypoints_with_scores.reshape(17, 3)
    print(keypoints_with_scores)
    print(reshaped)

    keypoint_coordinates = reshaped[reshaped[:, 2] > threshold]
    # keypoint_coordinates = keypoint_coordinates[:, [1, 0]]
    keypoint_coordinates[:, 0], keypoint_coordinates[:, 1] = keypoint_coordinates[:, 1], -keypoint_coordinates[:, 0]

    # Plot keypoints
    for keypoint, idx in KEYPOINT_DICT.items():
        plt.scatter(keypoint_coordinates[idx, 0], keypoint_coordinates[idx, 1], label=keypoint)

    # Connect keypoints with lines based on KEYPOINT_EDGE_INDS_TO_COLOR
    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        start_idx, end_idx = edge
        plt.plot([keypoint_coordinates[start_idx, 0], keypoint_coordinates[end_idx, 0]],
                 [keypoint_coordinates[start_idx, 1], keypoint_coordinates[end_idx, 1]],
                 color=color)

    # Set labels and show the plot
    plt.axis('off')
    if path:
        plt.savefig(f'{os.path.splitext(path)[0]}_stick.jpg', dpi=300, bbox_inches='tight')
    plt.show()


def get_keypoints_and_save_image(path, threshold=0.2):
    # change this path to try out different images
    keypoints = movenet(path)
    plot_keypoints_and_save(keypoints, threshold, path)
    return keypoints


get_keypoints_and_save_image('goddess_test_2.jpg')
