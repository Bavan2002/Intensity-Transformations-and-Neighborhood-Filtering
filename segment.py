import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/daisy.jpg'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # Load as RGB

# (a) GrabCut Segmentation
mask = np.zeros(image.shape[:2], np.uint8)  # Initial mask
bgd_model = np.zeros((1, 65), np.float64)  # Background model
fgd_model = np.zeros((1, 65), np.float64)  # Foreground model

# Define a rectangle around the flower (manually adjusted)
rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Convert the mask to binary: foreground pixels are set to 1, background to 0
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Extract the foreground and background images
foreground = image * mask2[:, :, np.newaxis]
background = image * (1 - mask2[:, :, np.newaxis])

# (b) Produce an enhanced image with a blurred background
blurred_background = cv2.GaussianBlur(image, (25, 25), 0)  # Substantial blur
enhanced_image = blurred_background.copy()
enhanced_image[mask2 == 1] = image[mask2 == 1]  # Replace the flower region with the sharp image

# Display the results
fig, ax = plt.subplots(2, 3, figsize=(18, 12))

# Original image
ax[0, 0].imshow(image)
ax[0, 0].set_title("Original Image")
ax[0, 0].axis("off")

# Segmentation mask
ax[0, 1].imshow(mask2, cmap='gray')
ax[0, 1].set_title("Segmentation Mask (Foreground)")
ax[0, 1].axis("off")

# Foreground image
ax[0, 2].imshow(foreground)
ax[0, 2].set_title("Foreground Image (Flower)")
ax[0, 2].axis("off")

# Background image
ax[1, 0].imshow(background)
ax[1, 0].set_title("Background Image (Without Flower)")
ax[1, 0].axis("off")

# Blurred background
ax[1, 1].imshow(blurred_background)
ax[1, 1].set_title("Blurred Background")
ax[1, 1].axis("off")

# Enhanced image with sharp foreground and blurred background
ax[1, 2].imshow(enhanced_image)
ax[1, 2].set_title("Enhanced Image with Blurred Background")
ax[1, 2].axis("off")

plt.show()

# (c) Explanation of darker edges around the flower
print("Explanation: The background just beyond the edge of the flower appears darker because when blurring is applied to the background, the pixels near the edge of the foreground (flower) are a blend of the background and the flower's colors. This blending causes the darker background colors to mix with the lighter colors of the flower, resulting in a darker appearance at the edges.")
