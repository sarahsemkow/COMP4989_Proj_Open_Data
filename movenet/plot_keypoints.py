import cv2
import numpy as np


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
