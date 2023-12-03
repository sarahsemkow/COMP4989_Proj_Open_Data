import numpy as np
import tensorflow as tf
import cv2

from constants import MOVENET_MODEL_PATH


class Movenet:
    def __init__(self, threshold=0.1):
        self.interpreter = tf.lite.Interpreter(model_path=MOVENET_MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.threshold = threshold
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_keypoints_with_scores(self, input_image):
        image = self.process_image(input_image)
        self.interpreter.set_tensor(self.input_details[0]["index"], np.array(image))
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self.output_details[0]["index"])
        keypoints_reshaped = keypoints.reshape(17, 3)
        return keypoints_reshaped

    def process_image(self, image_path):
        # Load the input image.
        image = tf.io.read_file(image_path)
        image = tf.compat.v1.image.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.image.resize_with_pad(image, 192, 192)
        # TF Lite format expects tensor type of float32.
        return tf.cast(image, dtype=tf.float32)

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        # Multiple keypoints to frame shape (no transformation applied to confidence)
        y, x, c = frame.shape
        # multiply the keypoints with the frame shape y, x, 1 *c (don't want to convert the confidence level)
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        # Loop through each keypoint (y,x,c) of the values in shaped
        for kp in shaped:
            # y-coord, x-coord, and confidence
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                size_of_circle = 6
                color_of_circle = (0, 255, 0)  # BGR
                circle_line_thickness = -1  # Thick line and fills in circle
                # Draw the dot on the image onto the frame that was passed in
                cv2.circle(
                    frame,
                    (int(kx), int(ky)),
                    size_of_circle,
                    color_of_circle,
                    circle_line_thickness,
                )
        return frame

    def draw_edges(self, frame, keypoints, edges, confidence_threshold):
        # get frame coordinates and mulitplies it to the keypoints so scaled properly
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            # make sure the confidence thresholds pass for both points
            if (c > confidence_threshold) & (c2 > confidence_threshold):
                color_of_line = (255, 0, 0)  # BGR
                line_thickness = 2
                # wrapping coordinates so that its just integer
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color_of_line,
                    line_thickness,
                )
        return frame
