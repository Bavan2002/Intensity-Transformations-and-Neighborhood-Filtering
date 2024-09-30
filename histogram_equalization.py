import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/shells.tif'
image = Image.open(image_path).convert('L')  # Convert image to grayscale
image_array = np.array(image)

# Function for histogram equalization
def histogram_equalization(image):
    # Compute the histogram of the image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Masking zeros in CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Histogram equalization formula
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Fill masked values with 0s
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply the transformation
    image_equalized = cdf[image]
    return image_equalized

# Apply histogram equalization to the image
image_equalized = histogram_equalization(image_array)

# Plot the original and equalized images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image_array, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(image_equalized, cmap='gray')
ax[1].set_title('Histogram Equalized Image')
ax[1].axis('off')

plt.show()

# Plot histograms before and after equalization
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].hist(image_array.flatten(), bins=256, range=(0, 256), color='black')
ax[0].set_title('Original Histogram')

ax[1].hist(image_equalized.flatten(), bins=256, range=(0, 256), color='black')
ax[1].set_title('Equalized Histogram')

plt.show()
