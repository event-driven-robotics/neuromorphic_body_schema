# test the animation code

import logging

import cv2
import numpy as np


x_start = []
y_start = []    

dt = 5 # simulation "time"

def move_line_left(previous_img, current_time , line_start_x, line_y, line_length=50, dt = 5):
    """
    Moves a line from its starting position towards the left by one pixel per frame.

    :param previous_img: The initial image with the line drawn at the starting position.
    :param line_start_x: The starting x-coordinate of the line.
    :param line_y: The y-coordinate of the line (constant for horizontal lines).
    :param line_length: The length of the line to be moved.
    :param n_frames: The number of frames over which the line will move.
    """
    # Make a copy of the previous image to avoid modifying the original
    # Copy the original image (reset each frame)
    current_img = previous_img.copy()

    for j in range(len(line_start_x)):
                    
        # Calculate the current x-coordinate of the line
        new_x = line_start_x[j] - current_time

        # Make sure the line doesn't move off the image
        if new_x < 0:
            new_x = 0

        # Draw the line on the image (this could be a horizontal or vertical line)
        # cv2.line(current_img, (new_x, line_y[j]), (new_x , line_y[j]+ line_length), (255, 255, 255), 2)
        cv2.circle(current_img, (new_x, line_y[j]), 10, (255, 255, 255), -1)

    # Display the image with the updated line position
    cv2.imshow("Moving Line", current_img)
    


    

# Example usage:
# Load your previous image (ensure it's a valid image)
previous_img = np.zeros((600, 800))

# Define the starting position and other parameters
# line_start_x = 300  # Starting x-coordinate of the line
# line_y = 300        # y-coordinate of the line
line_length = 20    # Length of the line
# n_frames = 300      # Number of frames to move the line

# Call the function to move the line
i=0
n_frames = 100
while i<n_frames: 
    
    x_new = 400 # np.random.randint(0, 800)
    y_new = 5*i   
    x_start.append(x_new)
    y_start.append(y_new)
    current_time = i*dt

    move_line_left(previous_img, current_time, x_start, y_start, line_length)
    i+=1
    # Wait a bit to show the update, adjust the delay if needed
    if cv2.waitKey(30) & 0xFF == ord('q'):  # press 'q' to exit the loop
        break

    # Close the OpenCV window after the loop
cv2.destroyAllWindows()

