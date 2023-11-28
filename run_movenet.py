# Import TF and TF Hub libraries.
import csv
import os

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    img = Image.open(image_path)
    print(img.mode)

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

    keypoint_coordinates = reshaped[reshaped[:, 2] > threshold]
    print("original:", reshaped.shape, f"\nreshaped with threshold at {threshold}:", keypoint_coordinates.shape)
    if reshaped[reshaped[:, 2] > threshold].shape[0] < 17:
        print("Not enough keypoints detected")
        return

    # keypoint_coordinates = keypoint_coordinates[:, [1, 0]]
    keypoint_coordinates[:, 0], keypoint_coordinates[:, 1] = keypoint_coordinates[:, 1], -keypoint_coordinates[:, 0]

    plt.figure(figsize=(5, 5))

    # Plot keypoints
    for keypoint, idx in KEYPOINT_DICT.items():
        plt.scatter(keypoint_coordinates[idx, 0], keypoint_coordinates[idx, 1], label=keypoint)

    # Connect keypoints with lines based on KEYPOINT_EDGE_INDS_TO_COLOR
    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        start_idx, end_idx = edge
        plt.plot([keypoint_coordinates[start_idx, 0], keypoint_coordinates[end_idx, 0]],
                 [keypoint_coordinates[start_idx, 1], keypoint_coordinates[end_idx, 1]],
                 color=color)

    # # Set labels and show the plot
    # plt.axis('off')
    # if path:
    #     new_root = os.path.dirname(path) + '_stick'
    #     os.makedirs(new_root, exist_ok=True)
    #     new_file_name = new_root + '/' + os.path.basename(path)
    #     plt.savefig(f'{new_file_name}', dpi=300, bbox_inches='tight')
    # # plt.legend()
    # plt.show()


def get_keypoints_and_save_image(path, threshold=0.2):
    # change this path to try out different images
    keypoints = movenet(path)
    # plot_keypoints_and_save(keypoints, threshold, path)
    return keypoints

def keypoint_coords_to_csv(movenet_keypoints):
    # Extract keypoints without confidence levels
    keypoints_filtered = [[keypoints[:2] for keypoints in sample.tolist()] for sample in movenet_keypoints]

    data = []
    for keypoints in keypoints_filtered:
        row = {}
        for i, keypoint in enumerate(KEYPOINT_DICT.keys()):
            row[f"{keypoint}_x"] = keypoints[i][0]
            row[f"{keypoint}_y"] = keypoints[i][1]
        data.append(row)
    # Specify the CSV file path
    csv_file_path = "keypoint_coordinates.csv"

    # Write data to CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = [f"{keypoint}_x" for keypoint in KEYPOINT_DICT.keys()] + [f"{keypoint}_y" for keypoint in
                                                                               KEYPOINT_DICT.keys()]
        interleaved_fieldnames = [field for pair in
                                  zip(fieldnames[:len(KEYPOINT_DICT)], fieldnames[len(KEYPOINT_DICT):]) for field in
                                  pair]
        writer = csv.DictWriter(csv_file, fieldnames=interleaved_fieldnames)
        # Write header
        writer.writeheader()
        for row in data:
            # Write keypoints data
            writer.writerow(row)
    print(f"Keypoints saved to {csv_file_path}")

keypoints = []
directory = './dataset/tree'
for index, filename in enumerate(os.listdir(directory)):
    # if filename.endswith(".jpg"):
    if filename.endswith(".DS_Store"):
        continue
    filepath = os.path.join(directory, filename)
    print(filepath)
    print(index)
    print((index / len(os.listdir(directory))) * 100, '% complete')
    keypoints.append(get_keypoints_and_save_image(filepath)[0][0])
keypoint_coords_to_csv(keypoints)

# keypoints_original = get_keypoints_and_save_image('./model/sample.jpg')
# print(keypoints_original)

# keypoints = get_keypoints_and_save_image('./dataset/downdog/242424242_327327.png')

# file_path_chosen = './dataset/downdog/242424242_315315.jpg'
# img = Image.open(file_path_chosen)
# print(img.mode)
