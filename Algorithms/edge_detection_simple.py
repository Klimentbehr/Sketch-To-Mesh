import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

def mark_corners(image, max_corners=100):
    """
    Detect and mark corners of shapes in an image with blue pins.
    :param image: Input image.
    :param max_corners: The maximum number of corners to detect.
    :return: Image with corners marked with blue dots, and array of corner coordinates.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    
    corner_coordinates = []  # Array to store corner coordinates
    
    # Refine the corner locations
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            corner_coordinates.append((x, y))  # Store corner coordinates
    
    return image, corner_coordinates

def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"The image at {image_path} could not be loaded. Please check the file path.")
        return

    # Mark corners with blue dots and get corner coordinates
    marked_image, corner_coordinates = mark_corners(image)

    # Display original and marked image using Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('Image with Corners Marked')
    ax2.axis('off')

    plt.show()

    return corner_coordinates

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    image_path = filedialog.askopenfilename(title="Select an image")
    if image_path:
        corner_coordinates = process_image(image_path)
        print("Corner Coordinates:")
        for corner in corner_coordinates:
            print(corner)
