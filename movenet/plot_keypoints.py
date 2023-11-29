import cv2
import numpy as np
from matplotlib import pyplot as plt

from constants import KEYPOINT_DICT, KEYPOINT_EDGE_INDS_TO_COLOR

# Define the keypoints' connection indices for drawing lines
keypoint_connections = KEYPOINT_EDGE_INDS_TO_COLOR

keypoints_list = [

        [0.24907833337783813, 0.517152726650238, 0.43927979469299316], [0.23474803566932678, 0.5245550870895386, 0.6269112825393677], [0.236936554312706, 0.5007495284080505, 0.5256597399711609], [0.247139573097229, 0.5462706089019775, 0.5676954388618469], [0.25226783752441406, 0.4844680726528168, 0.6927466988563538], [0.31631118059158325, 0.5776768326759338, 0.7648999691009521], [0.31990325450897217, 0.4676021933555603, 0.814132809638977], [0.19661518931388855, 0.6302960515022278, 0.633749783039093], [0.21072682738304138, 0.4067469537258148, 0.7963000535964966], [0.10235675424337387, 0.531099796295166, 0.3302854895591736], [0.10990302264690399, 0.48321378231048584, 0.387527734041214], [0.5401512384414673, 0.5495051145553589, 0.6794514060020447], [0.5368143320083618, 0.4697518050670624, 0.7218002080917358], [0.733249843120575, 0.5349695086479187, 0.5525133609771729], [0.6540764570236206, 0.34302622079849243, 0.9195476174354553], [0.8788710832595825, 0.5176581144332886, 0.607658863067627], [0.7001702189445496, 0.48595818877220154, 0.6411023139953613]

]

# Switching first and second values and making all values negative
switched_list = [[-val for val in [sublist[1], sublist[0], *sublist[2:]]] for sublist in keypoints_list]

# Displaying the modified list
print(switched_list)

# Function to plot keypoints
def plot_keypoints(keypoints, edges):
    plt.figure(figsize=(8, 8))
    plt.gca().set_facecolor('white')  # Set background color to white

    # Plot keypoints
    for kp in keypoints:
        plt.scatter(kp[0], kp[1], s=50, c='r', marker='o')  # Plotting keypoints as red circles

    # Plot edges
    for edge, color in edges.items():
        plt.plot([keypoints[edge[0]][0], keypoints[edge[1]][0]],
                 [keypoints[edge[0]][1], keypoints[edge[1]][1]],
                 color=color)

    plt.axis('off')
    plt.show()


# Plot keypoints and edges
plot_keypoints(switched_list, KEYPOINT_EDGE_INDS_TO_COLOR)

# # Example keypoints (replace with your keypoints)
# keypoints = np.array(keypoints_list)
#
# # Assuming the image dimensions (you can replace this with your image size)
# image_width, image_height = 640, 480
#
# # Create a blank image to draw the pose
# blank_image = np.zeros((image_height, image_width, 3), np.uint8)
# frame = np.array(keypoints_list)
#
# # Get the shape of the NumPy array
# # shape = array.shape
#
#
# # # Draw lines connecting keypoints
# for connection in keypoint_connections:
#     start_point = (int(keypoints[0][connection[0]][0] * image_width),
#                    int(keypoints[0][connection[0]][1] * image_height))
#     end_point = (int(keypoints[0][connection[1]][0] * image_width),
#                  int(keypoints[0][connection[1]][1] * image_height))
#     blank_image = cv2.line(blank_image, start_point, end_point, (0, 255, 0), 2)
#
# # Display the image with pose lines
# cv2.imshow('Pose', blank_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
