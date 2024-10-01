import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image

# Load the image
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/jeniffer.jpg'  # Replace with your image path if needed
image_rgb = np.array(Image.open(image_path))

# Convert the image to HSV (Hue, Saturation, Value) color space
image_hsv = color.rgb2hsv(image_rgb)

# Extract Hue, Saturation, and Value planes
hue, saturation, value = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]

# Display the HSV planes
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(hue, cmap='gray')
ax[0].set_title('Hue Plane')
ax[0].axis('off')

ax[1].imshow(saturation, cmap='gray')
ax[1].set_title('Saturation Plane')
ax[1].axis('off')

ax[2].imshow(value, cmap='gray')
ax[2].set_title('Value Plane')
ax[2].axis('off')

plt.show()

# Select the value plane for thresholding (to extract the foreground mask)
ret, mask = cv2.threshold((value * 255).astype('uint8'), 100, 255, cv2.THRESH_BINARY)

# Apply bitwise AND to extract only the foreground
foreground = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Convert foreground to grayscale for histogram equalization
foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)

# Apply histogram equalization to the grayscale foreground
foreground_equalized = cv2.equalizeHist(foreground_gray)

# Compute the cumulative sum of the histogram for reference
hist, bins = np.histogram(foreground_equalized.flatten(), 256, [0,256])
cdf = hist.cumsum()

# Display the original grayscale foreground and histogram-equalized foreground
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(foreground_gray, cmap='gray')
ax[0].set_title('Original Foreground (Grayscale)')
ax[0].axis('off')

ax[1].imshow(foreground_equalized, cmap='gray')
ax[1].set_title('Histogram Equalized Foreground')
ax[1].axis('off')

plt.show()

# Now recombine the histogram equalized foreground with the original background
background = cv2.bitwise_and(image_rgb, image_rgb, mask=cv2.bitwise_not(mask))
combined_image = cv2.add(background, cv2.cvtColor(foreground_equalized, cv2.COLOR_GRAY2RGB))

# Show the original image and final result with histogram equalized foreground
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(combined_image)
ax[1].set_title('Image with Histogram Equalized Foreground')
ax[1].axis('off')

plt.show()
