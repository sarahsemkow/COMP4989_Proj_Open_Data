import time
import cv2
import numpy as np
import tensorflow as tf

model_path = 'movenet_model/3.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

camera_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink"

# Open the camera
cap = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)


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

    # gets the input_details, and then sets the 'index' key to be the input_image
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    # make our predictions
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    cv2.imshow('MoveNet Lighting', frame)

    # get the key value
    key = cv2.waitKey(10) & 0xFF
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("failed to run webcam!")


