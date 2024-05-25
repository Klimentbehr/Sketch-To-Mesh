import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = r"C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\Cube_3_angle_36.00_30.00_POINT_1000.png"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours from the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
image_with_edges = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

# Display the original image and the image with edges
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Image with Edges')
plt.imshow(cv2.cvtColor(image_with_edges, cv2.COLOR_BGR2RGB))

plt.show()
