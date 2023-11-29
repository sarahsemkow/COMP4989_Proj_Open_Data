# Import TF and TF Hub libraries.
import cv2
import numpy as np
import tensorflow as tf
from constants import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR
from matplotlib import pyplot as plt, transforms

# https://www.youtube.com/watch?v=SSW9LzOJSus
# https://www.kaggle.com/models/google/movenet/frameworks/tfLite/variations/singlepose-lightning/versions/1?tfhub-redirect=true
''' singlepose-lighting '''


# frame = image, keypoints, confidence threshold
def draw_keypoints(frame, keypoints, confidence_threshold):
    # Multiple keypoints to frame shape (no transformation applied to confidence)
    y, x, c = frame.shape
    # multiply the keypoints with the frame shape y, x, 1 *c (don't want to convert the confidence level)
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    # Loop through each keypoint (y,x,c) of the values in shaped
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            size_of_circle = 4
            color_of_circle = (0, 255, 0)  # BGR
            circle_line_thickness = -1  # Thick line and fills in circle
            # Draw the dot in the image
            cv2.circle(frame, (int(kx), int(ky)), size_of_circle, color_of_circle, circle_line_thickness)
    return frame

    # Define connections between keypoints (edges)
    # connections = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
    #                (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    # color_of_line = (255, 0, 0)  # BGR color for lines (blue)
    # line_thickness = 2  # Line thickness for connections
    #
    # for connection in connections:
    #     kp1, kp2 = connection
    #     ky1, kx1, kp1_conf = shaped[kp1]
    #     ky2, kx2, kp2_conf = shaped[kp2]
    #     if kp1_conf > confidence_threshold and kp2_conf > confidence_threshold:
    #         # (image, center_coordinates, radius, color, thickness)
    #         cv2.line(frame, (int(kx1), int(ky1)), (int(kx2), int(ky2)), color_of_line, line_thickness)
    # return frame


def draw_edges(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return frame


# Load the input image.
image_path = 'images/tree_1.jpg'
# image = tf.io.read_file(image_path)
# image = tf.compat.v1.image.decode_jpeg(image)
image = cv2.imread(image_path)
frame = image.copy()  # Copy the image to 'frame' variable

# encapsulate inside another array
image = tf.expand_dims(image, axis=0)
# resize because movenet needs the image to be 192x192
image = tf.image.resize_with_pad(image, 192, 192)

''' Load model, intialize the TFLite interpreter '''
model_path = 'movenet_model/3.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# TF Lite format expects tensor type of float32.
input_image = tf.cast(image, dtype=tf.float32)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

''' Make predictions '''
# set input details {index:0} as input image that was reshaped and converted to float (tensor requires)
interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
# invoke the prediction
interpreter.invoke()
# output_details[0]['index'] = nparray, get_tensor gets the result back out
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
# print(keypoints_with_scores)

interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

interpreter.invoke()

# Output is a [1, 1, 17, 3] numpy array.
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
print(np.squeeze(keypoints_with_scores).tolist())

''' change image to frames to pass to the function '''
frame_shape = frame.shape  # Get the shape of the frame
print(type(frame_shape))
frame = cv2.resize(frame, (192, 192))

''' Rendering section '''
result_image = draw_keypoints(frame, keypoints_with_scores, 0.1)
result_image = draw_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.1)
cv2.imshow('result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#######
# size_of_circle = 5
# color_of_circle = (0, 255, 0)  # BGR
# circle_line_thickness = -1  # Thick line and fills in circle
# # Draw the dot in the image
# cv2.circle(image, (120, 120), size_of_circle, color_of_circle, circle_line_thickness)
# cv2.imshow('sup', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
