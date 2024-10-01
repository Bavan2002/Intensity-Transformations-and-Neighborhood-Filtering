import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03small.png'  # Replace with your image path if needed
image = np.array(Image.open(image_path).convert('RGB'))

# (a) Nearest-Neighbor Interpolation
def nearest_neighbor_zoom(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Create a grid of coordinates in the output image
    row_indices = (np.arange(new_height) / scale_factor).astype(int)
    col_indices = (np.arange(new_width) / scale_factor).astype(int)

    # Clip indices to ensure they stay within the bounds of the original image
    row_indices = np.clip(row_indices, 0, height - 1)
    col_indices = np.clip(col_indices, 0, width - 1)

    # Use advanced indexing to map the original image to the zoomed image
    zoomed_image = image[row_indices[:, None], col_indices]

    return zoomed_image

# (b) Bilinear Interpolation
def bilinear_interpolation_zoom(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Create grid of output coordinates
    row_indices = np.linspace(0, height - 1, new_height)
    col_indices = np.linspace(0, width - 1, new_width)

    # Get integer and fractional parts
    row_floor = np.floor(row_indices).astype(int)
    row_ceil = np.minimum(row_floor + 1, height - 1)
    row_fraction = row_indices - row_floor

    col_floor = np.floor(col_indices).astype(int)
    col_ceil = np.minimum(col_floor + 1, width - 1)
    col_fraction = col_indices - col_floor

    # Initialize zoomed image
    zoomed_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # Apply bilinear interpolation
    for i in range(channels):
        top_left = image[row_floor[:, None], col_floor, i]
        top_right = image[row_floor[:, None], col_ceil, i]
        bottom_left = image[row_ceil[:, None], col_floor, i]
        bottom_right = image[row_ceil[:, None], col_ceil, i]

        top = top_left * (1 - col_fraction) + top_right * col_fraction
        bottom = bottom_left * (1 - col_fraction) + bottom_right * col_fraction
        zoomed_image[:, :, i] = top * (1 - row_fraction[:, None]) + bottom * row_fraction[:, None]

    return zoomed_image


# Sum of Squared Differences (SSD)
def compute_ssd(image1, image2):
    # Ensure images have the same dimensions by resizing image2 to image1's size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Compute SSD
    ssd_value = np.sum((image1.astype('float') - image2.astype('float')) ** 2)
    normalized_ssd = ssd_value / image1.size  # Normalized by the number of pixels
    return normalized_ssd

# Zoom the image by a factor of 4 using nearest-neighbor and bilinear interpolation
zoom_factor = 4
zoomed_nn = nearest_neighbor_zoom(image, zoom_factor)
zoomed_bilinear = bilinear_interpolation_zoom(image, zoom_factor)


# Display the zoomed images
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(zoomed_nn)
axs[1].set_title(f'Zoomed (Nearest-Neighbor) x{zoom_factor}')
axs[1].axis('off')

axs[2].imshow(zoomed_bilinear)
axs[2].set_title(f'Zoomed (Bilinear Interpolation) x{zoom_factor}')
axs[2].axis('off')

plt.show()

downscaled_image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03small.png'  # Replace with your image path if needed
downscaled_image = np.array(Image.open(image_path).convert('RGB'))

original_image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03.png'  # Replace with your image path if needed
original_image = np.array(Image.open(image_path).convert('RGB'))


# Now scale it back up by factor of 4 using both methods
upsampled_nn = nearest_neighbor_zoom(downscaled_image, zoom_factor)
upsampled_bilinear = bilinear_interpolation_zoom(downscaled_image, zoom_factor)

# Compute SSD between the original image and upsampled images
ssd_nn = compute_ssd(original_image, upsampled_nn)
ssd_bilinear = compute_ssd(original_image, upsampled_bilinear)

# Output the SSD results
print(f"SSD (Nearest-Neighbor): {ssd_nn}")
print(f"SSD (Bilinear Interpolation): {ssd_bilinear}")
