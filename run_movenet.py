# Import TF and TF Hub libraries.
import argparse
import csv
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

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


def movenet(image_path, threshold):
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

    keypoints_with_scores = keypoints_with_scores.reshape(17, 3)
    keypoint_coordinates_within_threshold = keypoints_with_scores[keypoints_with_scores[:, 2] > threshold]
    if keypoint_coordinates_within_threshold.shape[0] < 17:
        print("Not enough keypoints detected")
        return
    return keypoints_with_scores


def keypoint_coords_to_csv(movenet_keypoints):
    keypoints_filtered = movenet_keypoints[:, :2]
    # keypoints_filtered[:, 0], keypoints_filtered[:, 1] = keypoints_filtered[:, 1], -keypoints_filtered[:, 0]
    keypoints_filtered = keypoints_filtered.tolist()

    data = []
    for keypoints in keypoints_filtered:
        print(keypoints)
        row = {}
        for i, keypoint in enumerate(KEYPOINT_DICT.keys()):
            # row[f"{keypoint}_x"] = keypoints[0]
            # row[f"{keypoint}_y"] = keypoints[1]
            row[f"{keypoint}_x"] = keypoints[1]
            row[f"{keypoint}_y"] = keypoints[0]
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

# keypoints = []
# directory = './dataset/tree'
# for index, filename in enumerate(os.listdir(directory)):
#     # if filename.endswith(".jpg"):
#     if filename.endswith(".DS_Store"):
#         continue
#     filepath = os.path.join(directory, filename)
#     print(filepath)
#     print(index)
#     print((index / len(os.listdir(directory))) * 100, '% complete')
#     keypoints.append(get_keypoints_and_save_image(filepath)[0][0])



def main():
    parser = argparse.ArgumentParser(description='Write data to a CSV file.')
    parser.add_argument('filepath', type=str, help='Path to image file')
    parser.add_argument('threshold', type=float, help='Threshold for keypoint detection')

    args = parser.parse_args()

    keypoints = movenet(args.filepath, args.threshold)
    # print(keypoints)
    keypoint_coords_to_csv(keypoints)


if __name__ == '__main__':
    main()

# Write keypoints to csv
# df = pd.DataFrame(keypoints)
# df.to_csv('keypoints.csv', index=False)
