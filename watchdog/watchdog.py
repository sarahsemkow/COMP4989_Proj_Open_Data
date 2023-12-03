import subprocess
import os
import time


def capture_image(image_directory):
    """Capture image from webcam and save it to folder"""
    cmd = f"nvgstcapture-1.0 --mode=1 --automate --capture-auto --file-name='./{image_directory}/yoga_pose'"
    try:
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode('utf-8')
        print(f"Error output: {error_output}")


def get_new_image_name(directory, image_directory):
    before_capture = set(os.listdir(directory))

    capture_image(image_directory)

    time.sleep(3)  # Takes time to save image

    after_capture = set(os.listdir(directory))

    new_image = after_capture - before_capture

    if len(new_image) == 1:
        return new_image.pop()

    print("No new image found")
    return None


def delete_image(image_name):
    """Delete image"""
    try:
        os.remove(image_name)
    except FileNotFoundError:
        print(f"File {image_name} not found")


def main():
    image_directory = "captured_image"
    base_directory = os.getcwd()
    directory = os.path.join(base_directory, image_directory)

    while True:
        image_captured = get_new_image_name(directory, image_directory)
        print(image_captured)

        if image_captured:
            delete_image(os.path.join(directory, image_captured))


if __name__ == "__main__":
    main()
