import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# Load the image
image_path = '/path_to_your_image/image.png'  # Replace with your image path
image_rgb = np.array(Image.open(image_path))

# Convert the image to HSV (Hue, Saturation, Value) color space
image_hsv = color.rgb2hsv(image_rgb)

# Extract the saturation plane
hue, saturation, value = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]

# Parameters for the intensity transformation
sigma = 70
a = 0.7  # You can adjust this value for visual appeal

# Apply the intensity transformation to the saturation plane
def vibrance_intensity_transformation(x, a, sigma=70):
    return np.minimum(x + a * 128 * np.exp(-((x * 255 - 128)**2) / (2 * sigma**2)), 255) / 255

saturation_transformed = vibrance_intensity_transformation(saturation * 255, a, sigma)

# Recombine the three planes (hue, transformed saturation, and value) into the HSV image
image_hsv_transformed = np.stack((hue, saturation_transformed, value), axis=2)

# Convert the transformed HSV image back to RGB
image_rgb_transformed = color.hsv2rgb(image_hsv_transformed)

# Display the original and vibrance-enhanced images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(image_rgb_transformed)
ax[1].set_title(f'Vibrance Enhanced Image (a = {a})')
ax[1].axis('off')

plt.show()

# Display the transformation applied to the saturation plane
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(saturation, cmap='gray')
ax[0].set_title('Original Saturation Plane')
ax[0].axis('off')

ax[1].imshow(saturation_transformed, cmap='gray')
ax[1].set_title('Transformed Saturation Plane')
ax[1].axis('off')

plt.show()
