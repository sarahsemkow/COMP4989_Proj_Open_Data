import time
import cv2
import numpy as np
import tensorflow as tf
from constants import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR
from plot_keypoints import *


# https://www.youtube.com/watch?v=SSW9LzOJSus
model_path = 'movenet_model/3.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# connects to webcam (can also pass in video here: 'video.mp4'/play around with 0)
cap = cv2.VideoCapture(0)

# timer that can takes screenshot every X seconds
start_time = time.time()
# set the number of second intervals
interval = 10
# image count (if taking screenshots)
ss_count = 1

# loop through every single frame in webcam
while cap.isOpened():
    # ret = whether a frame was successfully read or not (success status)
    # frame = frame that was read, image represented as arrays [h, w, 3channels]
    ret, frame = cap.read()

    ''' reshape image '''
    # copy the frame into an image
    img = frame.copy()
    # reshape image, movenet only takes in 192x192 and float32
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    ''' setup input and output (part of working with tflite) - see below how it looks like'''
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ''' make the predictions for the keypoints '''
    # gets the input_details, and then sets the 'index' key to be the input_image
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    # print(keypoints_with_scores)

    ''' render '''
    draw_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.2)
    draw_keypoints(frame, keypoints_with_scores, 0.2)

    # whats captured by webcam
    cv2.imshow('MoveNet Lighting', frame)

    # Check if it's time to take a screenshot
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= interval:
        # Save the screenshot as an image file/ to replace previous one, just remove {ss_count}
        # cv2.imwrite(f'screenshot{ss_count}.png', frame)
        print(keypoints_with_scores)
        # Increment the screenshot count and reset the timer
        ss_count += 1
        start_time = time.time()

    # get the key value
    key = cv2.waitKey(10) & 0xFF
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



'''
print(interpreter.get_input_details())
[{'name': 'serving_default_input:0', 
'index': 0, <<<<<<<<<<<<<<<<<<<<<<<<< input image placed here
'shape': array([  1, 192, 192,   3]), 
'shape_signature': array([  1, 192, 192,   3]), 
'dtype': <class 'numpy.float32'>, 
'quantization': (0.0, 0), 
'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 
'sparsity_parameters': {}}]
'''

