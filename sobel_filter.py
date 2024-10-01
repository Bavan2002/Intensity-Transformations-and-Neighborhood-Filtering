import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image in grayscale
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/einstein.png'  # Replace with your image path if needed
image = Image.open(image_path).convert('L')  # Convert image to grayscale
image_np = np.array(image)

# (a) Using OpenCV filter2D function for Sobel filtering
# Define Sobel kernels for x and y direction
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Apply Sobel filters using cv2.filter2D
sobel_x_filtered = cv2.filter2D(image_np, -1, sobel_x)
sobel_y_filtered = cv2.filter2D(image_np, -1, sobel_y)

# Compute the gradient magnitude
sobel_magnitude = np.sqrt(sobel_x_filtered**2 + sobel_y_filtered**2)

# Normalize the result
sobel_magnitude = np.uint8(sobel_magnitude / np.max(sobel_magnitude) * 255)

# (b) Write your own code to perform Sobel filtering (Manual convolution)
def sobel_filter_manual(image, kernel):
    # Get the dimensions of the image and kernel
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    
    # Output image initialized to zeros
    output = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Pad the image to handle the border effects
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    
    # Convolution operation
    for i in range(image_height):
        for j in range(image_width):
            output[i, j] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width])
    
    return output

# Apply manual Sobel filters
sobel_x_manual = sobel_filter_manual(image_np, sobel_x)
sobel_y_manual = sobel_filter_manual(image_np, sobel_y)

# Compute the gradient magnitude for manual method
sobel_magnitude_manual = np.sqrt(sobel_x_manual**2 + sobel_y_manual**2)

# Normalize the result
sobel_magnitude_manual = np.uint8(sobel_magnitude_manual / np.max(sobel_magnitude_manual) * 255)

# (c) Using the property of Sobel filter decomposition
sobel_1 = np.array([[1], [2], [1]])  # Vertical part
sobel_2 = np.array([[1, 0, -1]])     # Horizontal part

# Perform separable filtering
sobel_x_separable = cv2.sepFilter2D(image_np, -1, sobel_2, sobel_1)
sobel_y_separable = cv2.sepFilter2D(image_np, -1, sobel_1.T, sobel_2.T)

# Compute the gradient magnitude
sobel_magnitude_separable = np.sqrt(sobel_x_separable**2 + sobel_y_separable**2)
sobel_magnitude_separable = np.uint8(sobel_magnitude_separable / np.max(sobel_magnitude_separable) * 255)

# Display the results
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(sobel_x_filtered, cmap='gray')
axs[0, 0].set_title('Sobel X (filter2D)')
axs[0, 0].axis('off')

axs[0, 1].imshow(sobel_y_filtered, cmap='gray')
axs[0, 1].set_title('Sobel Y (filter2D)')
axs[0, 1].axis('off')

axs[0, 2].imshow(sobel_magnitude, cmap='gray')
axs[0, 2].set_title('Gradient Magnitude (filter2D)')
axs[0, 2].axis('off')

axs[1, 0].imshow(sobel_x_manual, cmap='gray')
axs[1, 0].set_title('Sobel X (Manual)')
axs[1, 0].axis('off')

axs[1, 1].imshow(sobel_y_manual, cmap='gray')
axs[1, 1].set_title('Sobel Y (Manual)')
axs[1, 1].axis('off')

axs[1, 2].imshow(sobel_magnitude_manual, cmap='gray')
axs[1, 2].set_title('Gradient Magnitude (Manual)')
axs[1, 2].axis('off')

plt.show()

# Display the separable Sobel results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(sobel_x_separable, cmap='gray')
ax[0].set_title('Sobel X (Separable)')
ax[0].axis('off')

ax[1].imshow(sobel_y_separable, cmap='gray')
ax[1].set_title('Sobel Y (Separable)')
ax[1].axis('off')

ax[2].imshow(sobel_magnitude_separable, cmap='gray')
ax[2].set_title('Gradient Magnitude (Separable)')
ax[2].axis('off')

plt.show()
