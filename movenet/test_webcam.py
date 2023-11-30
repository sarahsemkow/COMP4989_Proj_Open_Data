import time
import cv2
import tensorflow as tf

# model_path = 'movenet_model/3.tflite'
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# connects to webcam (can also pass in video here: 'video.mp4'/play around with 0)
cap = cv2.VideoCapture(0)

# timer that can takes screenshot every X seconds
start_time = time.time()
# set the number of second intervals
interval = 10
# image count
ss_count = 1

# loop through every single frame in webcam
while cap.isOpened():
    # ret = whether a frame was successfully read or not (success status)
    # frame = frame that was read, image represented as arrays [480H, 640W, 3channels]
    ret, frame = cap.read()

    # pass in the frame we want to render, this shows whats captured by webcam
    cv2.imshow('MoveNet Lighting', frame)

    # Check if it's time to take a screenshot
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time >= interval:
        # Save the screenshot as an image file/ to replace previous one, just remove {ss_count}
        cv2.imwrite(f'screenshot{ss_count}.png', frame)
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


