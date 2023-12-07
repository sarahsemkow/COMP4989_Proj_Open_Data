import time
import cv2
import numpy as np
import tensorflow as tf

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


def draw_keypoints(frame, keypoints, confidence_threshold):
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
            # Draw the dot on the image onto the frame that waas passed in
            cv2.circle(frame, (int(kx), int(ky)), size_of_circle, color_of_circle, circle_line_thickness)
    return frame


def draw_edges(frame, keypoints, edges, confidence_threshold):
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
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_of_line, line_thickness)
    return frame


model_path = 'movenet_model/3.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

camera_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink"

# Open the camera
# cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(0)

# loop through every single frame in webcam
while cap.isOpened():
    # ret = whether a frame was successfully read or not (success status)
    # frame = frame that was read, image represented as arrays [480H, 640W, 3channels]
    ret, frame = cap.read()

    # reshape image, movenet only takes in 192x192
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # setup input and output (part of working with tflite)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    """ make the predictions for the keypoints """
    # gets the input_details, and then sets the 'index' key to be the input_image
    interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
    keypoints_with_scores = keypoints_with_scores.reshape(17, 3)
    # print(keypoints_with_scores)

    """ render """
    draw_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.2)
    draw_keypoints(frame, keypoints_with_scores, 0.2)

    cv2.imshow('MoveNet Lighting', frame)

    # get the key value
    key = cv2.waitKey(10) & 0xFF
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("failed to run webcam!")


