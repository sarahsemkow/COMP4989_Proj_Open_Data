# Import TF and TF Hub libraries.
import csv
import os

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from constants import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR, ANGLE_DICT, LIMB_DICT


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

def keypoint_coords_to_csv(movenet_keypoints, file_name):
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
    csv_file_path = file_name

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

def calculate_angle(keypoint1, keypoint2, keypoint3):
    # Calculate vectors for the two lines
    vector1 = np.array(keypoint1) - np.array(keypoint2)
    vector2 = np.array(keypoint3) - np.array(keypoint2)

    # Calculate dot product
    dot_product = np.dot(vector1, vector2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def extract_keypoint_coords(row, keypoints):
    coords = []
    for keypoint in keypoints:
        x_col = f"{keypoint}_x"
        y_col = f"{keypoint}_y"
        coords.append([row[x_col], row[y_col]])
    return coords
# Function to calculate angle given keypoint indices
def calculate_angle(row, angle_dict):
    angle_results = {}
    for angle_key, keypoints in angle_dict.items():
        keypoint_coords = [extract_keypoint_coords(row, [keypoint]) for keypoint in keypoints]
        keypoint_coords_flattened = [x[0] for x in keypoint_coords]
        angle = calculate_angle_from_coords(*keypoint_coords_flattened)
        angle_results[angle_key] = angle
    return angle_results

# Function to calculate angle from coordinates
def calculate_angle_from_coords(coord1, coord2, coord3):
    vector1 = np.array(coord1) - np.array(coord2)
    vector3 = np.array(coord3) - np.array(coord2)

    dot_product = np.dot(vector1, vector3)
    magnitude1 = np.linalg.norm(vector1)
    magnitude3 = np.linalg.norm(vector3)

    cos_angle = dot_product / (magnitude1 * magnitude3)
    # Ensure the cosine value is within the valid range
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Step 1, obtain keypoint coordinates from images
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

coord_file_name = "keypoint_coordinates.csv"
keypoint_coords_to_csv(keypoints, coord_file_name)

# Step 2, calculate angles from keypoints
df = pd.read_csv(coord_file_name)
# Apply the calculate_angle function to each row of the DataFrame
angle_series = df.apply(calculate_angle, axis=1, angle_dict=ANGLE_DICT)
angle_df = pd.DataFrame.from_dict(angle_series.to_dict(), orient='index')
# Concatenate the angle results to the original DataFrame
df_with_angles = pd.concat([df, angle_df], axis=1)
df_with_angles.to_csv("keypoints_with_angles.csv", index=False)





# keypoints_original = get_keypoints_and_save_image('./model/sample.jpg')
# print(keypoints_original)

# keypoints = get_keypoints_and_save_image('./dataset/downdog/242424242_327327.png')

# file_path_chosen = './dataset/downdog/242424242_315315.jpg'
# img = Image.open(file_path_chosen)
# print(img.mode)
