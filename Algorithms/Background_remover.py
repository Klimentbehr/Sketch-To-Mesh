import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def select_image():
    # Set up the root window but don't display it
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

def remove_background(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Image not found")
        return

    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the range of colors to mask
    # For example, keeping only white-ish areas; adjust as needed
    lower = np.array([200, 200, 200], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")

    # Create a mask using the color range
    mask = cv2.inRange(image_rgb, lower, upper)

    # Create an inverse mask to keep the area of the original image
    mask_inv = cv2.bitwise_not(mask)

    # Convert original image to RGBA (so it supports transparency)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Use the inverse mask to retain the part of the image with the original color
    image_rgba[:, :, 3] = cv2.bitwise_and(mask_inv, image_rgba[:, :, 3])

    # Save the image with a transparent background
    cv2.imwrite(output_image_path, image_rgba)

    print(f"Output saved as {output_image_path}")

# Ask the user to select an image
input_image_path = select_image()

if input_image_path:
    output_image_path = input_image_path.rsplit('.', 1)[0] + "_no_bg.png"
    remove_background(input_image_path, output_image_path)
else:
    print("No file selected.")
