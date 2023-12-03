# Import TF and TF Hub libraries.
import csv

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from constants import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR


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
    model_path = "../models/3.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
    interpreter.invoke()

    # Output is a [1, 1, 17, 3] numpy array.
    # Keypoint coordinates are in the range [0.0, 1.0] normalized by the input image size.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])

    keypoints_with_scores = keypoints_with_scores.reshape(17, 3)
    keypoint_coordinates_within_threshold = keypoints_with_scores[
        keypoints_with_scores[:, 2] > threshold
    ]
    if keypoint_coordinates_within_threshold.shape[0] < 17:
        print("Not enough keypoints detected")
        return
    return keypoints_with_scores


def plot_keypoints(keypoints_with_scores, threshold):
    reshaped = keypoints_with_scores.reshape(17, 3)

    keypoint_coordinates = reshaped[reshaped[:, 2] > threshold]
    print(
        "original:",
        reshaped.shape,
        f"\nreshaped with threshold at {threshold}:",
        keypoint_coordinates.shape,
    )
    if reshaped[reshaped[:, 2] > threshold].shape[0] < 17:
        print("Not enough keypoints detected")
        return

    # keypoint_coordinates = keypoint_coordinates[:, [1, 0]]
    # keypoint_coordinates[:, 0], keypoint_coordinates[:, 1] = keypoint_coordinates[:, 1], -keypoint_coordinates[:, 0]

    plt.figure(figsize=(5, 5))

    # Plot keypoints
    for keypoint, idx in KEYPOINT_DICT.items():
        plt.scatter(
            keypoint_coordinates[idx, 0], keypoint_coordinates[idx, 1], label=keypoint
        )

    # Connect keypoints with lines based on KEYPOINT_EDGE_INDS_TO_COLOR
    for edge, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        start_idx, end_idx = edge
        plt.plot(
            [keypoint_coordinates[start_idx, 0], keypoint_coordinates[end_idx, 0]],
            [keypoint_coordinates[start_idx, 1], keypoint_coordinates[end_idx, 1]],
            color=color,
        )

    # Set labels and show the plot
    # plt.axis('off')
    # if path:
    # new_root = os.path.dirname(path) + '_stick'
    # os.makedirs(new_root, exist_ok=True)
    # new_file_name = new_root + '/' + os.path.basename(path)
    # plt.savefig(f'{new_file_name}', dpi=300, bbox_inches='tight')
    plt.legend()
    plt.show()
    # plt.close()


def keypoint_coords_to_csv(movenet_keypoints):
    keypoints_filtered = movenet_keypoints[:, :2]
    # keypoints_filtered[:, 0], keypoints_filtered[:, 1] = keypoints_filtered[:, 1], -keypoints_filtered[:, 0]
    keypoints_filtered = keypoints_filtered.tolist()

    plot_keypoints(movenet_keypoints, 0.1)

    data = []
    for keypoints in keypoints_filtered:
        print(keypoints)
        row = {}
        for i, keypoint in enumerate(KEYPOINT_DICT.keys()):
            row[f"{keypoint}_x"] = keypoints[0]
            row[f"{keypoint}_y"] = keypoints[1]
        data.append(row)
    # Specify the CSV file path
    csv_file_path = "keypoint_coordinatesss.csv"

    # Write data to CSV
    with open(csv_file_path, mode="w", newline="") as csv_file:
        fieldnames = [f"{keypoint}_x" for keypoint in KEYPOINT_DICT.keys()] + [
            f"{keypoint}_y" for keypoint in KEYPOINT_DICT.keys()
        ]
        interleaved_fieldnames = [
            field
            for pair in zip(
                fieldnames[: len(KEYPOINT_DICT)], fieldnames[len(KEYPOINT_DICT) :]
            )
            for field in pair
        ]
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
    # parser = argparse.ArgumentParser(description='Write data to a CSV file.')
    # parser.add_argument('filepath', type=str, help='Path to image file')
    # parser.add_argument('threshold', type=float, help='Threshold for keypoint detection')

    # args = parser.parse_args()

    # keypoints = movenet(args.filepath, args.threshold)
    filepath = "./dataset/tree/242424242_440440.jpg"
    threshold = 0.1
    keypoints = movenet(filepath, threshold)
    # print(keypoints)
    keypoint_coords_to_csv(keypoints)


if __name__ == "__main__":
    main()
