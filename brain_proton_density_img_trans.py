import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the brain proton density image in grayscale
brain_image = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice.png', cv2.IMREAD_GRAYSCALE)

# Display the original brain image
plt.imshow(brain_image, cmap='gray')
plt.title("Original Brain Proton Density Image")
plt.show()

# Output the shape of the image
print(brain_image.shape)

# Coordinates for sampling intensities
white_matter_coords = (130, 110)
gray_matter_coords = (140, 90)

# Show locations of white and gray matter in the image
plt.imshow(brain_image, cmap="gray")
plt.scatter(white_matter_coords[0], white_matter_coords[1], color='red', label='White Matter')
plt.scatter(gray_matter_coords[0], gray_matter_coords[1], color='blue', label='Gray Matter')
plt.legend()
plt.show()

# Retrieve pixel intensities at specified points
white_matter_intensity = brain_image[white_matter_coords]
gray_matter_intensity = brain_image[gray_matter_coords]

# Output pixel intensities for white and gray matter
print(f"White Matter Intensity: {white_matter_intensity}")
print(f"Gray Matter Intensity: {gray_matter_intensity}")

# Function to apply intensity transformations for highlighting white and gray matter
def transform_intensities(image):
    # Copy the input image for transformation
    transformed_img = np.copy(image)
    
    # Apply transformation for gray matter intensities (186 to 250)
    gray_matter_mask = (image >= 186) & (image <= 250)
    transformed_img[gray_matter_mask] = 1.75 * image[gray_matter_mask] + 30
    
    # Apply transformation for white matter intensities (150 to 185)
    white_matter_mask = (image >= 150) & (image <= 185)
    transformed_img[white_matter_mask] = 1.55 * image[white_matter_mask] + 22.5
    
    return transformed_img, white_matter_mask, gray_matter_mask

# Apply the transformation to the brain image
transformed_img, white_matter_mask, gray_matter_mask = transform_intensities(brain_image)

# Display the transformed image with enhanced white and gray matter
plt.imshow(transformed_img, cmap='gray')
plt.title("Transformed Brain Image: White and Gray Matter Enhanced")
plt.show()

# Display the white matter mask
plt.imshow(white_matter_mask, cmap='gray')
plt.title("White Matter")
plt.show()

# Display the gray matter mask
plt.imshow(gray_matter_mask, cmap='gray')
plt.title("Gray Matter")
plt.show()

# Create intensity transformation curves for both white and gray matter
x_vals = np.arange(0, 256)  # Intensity range (0-255)
gray_transformed = np.array([1.75 * x + 30 if 186 <= x <= 250 else x for x in x_vals])
white_transformed = np.array([1.55 * x + 22.5 if 150 <= x <= 185 else x for x in x_vals])

# Plot the transformation curves for both white and gray matter
plt.figure(figsize=(10, 6))
plt.plot(x_vals, white_transformed, label='White Matter Transformation', color='blue')
plt.plot(x_vals, gray_transformed, label='Gray Matter Transformation', color='green')
plt.plot(x_vals, x_vals, label='Original Intensity', linestyle='--', color='red')  # Identity line for reference
plt.title('Intensity Transformation Curves for White and Gray Matter')
plt.xlabel('Original Intensity')
plt.ylabel('Transformed Intensity')
plt.legend()
plt.grid(True)
plt.show()

