import cv2
import time

camera_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink"

# Open the camera
camera = cv2.VideoCapture(camera_pipeline, cv2.CAP_GSTREAMER)

# Capture and save 10 images with a 1-second interval
for i in range(10):
    # Read a frame
    ret, image = camera.read()

    if ret:
        # Save the image
        cv2.imwrite(f"captured_image_{i + 1}.jpg", image)
        print(f"Image {i + 1} captured.")
    else:
        print(f"Error capturing image {i + 1}.")

    # Wait for 1 second
    time.sleep(1)

# Release the camera
camera.release()