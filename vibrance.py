import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# Load the image
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/spider.png'  # Replace with your image path if needed
image_rgb = np.array(Image.open(image_path)) / 255.0  # Normalized to [0, 1] for skimage

# Convert the image to HSV (Hue, Saturation, Value) color space
image_hsv = color.rgb2hsv(image_rgb)

# Extract the hue, saturation, and value planes
hue, saturation, value = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]

# Parameters for the intensity transformation
sigma = 70
a = 1.5# You can adjust this value to get the desired vibrance effect

# Apply the intensity transformation to the saturation plane
def vibrance_intensity_transformation(x, a, sigma=70):
    return np.minimum(x + a * 128 * np.exp(-((x * 255 - 128)**2) / (2 * sigma**2)), 255) / 255

# Apply the transformation to the saturation channel
saturation_transformed = vibrance_intensity_transformation(saturation, a, sigma)

# Recombine the transformed saturation plane with the original hue and value planes
image_hsv_transformed = np.stack((hue, saturation_transformed, value), axis=2)

# Convert the transformed HSV image back to RGB
image_rgb_transformed = color.hsv2rgb(image_hsv_transformed)

# Plot the original and vibrance-enhanced images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(image_rgb_transformed)
ax[1].set_title(f'Vibrance Enhanced Image (a = {a})')
ax[1].axis('off')

plt.show()

# Plot the saturation planes before and after transformation
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(saturation, cmap='gray')
ax[0].set_title('Original Saturation Plane')
ax[0].axis('off')

ax[1].imshow(saturation_transformed, cmap='gray')
ax[1].set_title('Transformed Saturation Plane')
ax[1].axis('off')

plt.show()

# Generate a range of intensity values (0 to 255)
x = np.linspace(0, 255, 256)
y = vibrance_intensity_transformation(x, a, sigma)

# Plot the intensity transformation
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=f'Intensity Transformation (a = {a}, sigma = {sigma})', color='blue')
plt.title('Intensity Transformation Function')
plt.xlabel('Input Intensity (x)')
plt.ylabel('Transformed Intensity (f(x))')
plt.grid(True)
plt.legend()
plt.show()

