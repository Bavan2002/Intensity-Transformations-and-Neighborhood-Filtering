import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the image
spiderman_image = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/spider.png')

# Convert the image to HSV color space
hsv_spiderman_image = cv2.cvtColor(spiderman_image, cv2.COLOR_BGR2HSV)

# Display shape of the image in HSV space
print(hsv_spiderman_image.shape)

# Display the hue channel of the HSV image
plt.imshow(hsv_spiderman_image[:, :, 0], cmap="gray")
plt.show()

# Extract the individual H, S, and V planes
hue_channel = hsv_spiderman_image[:, :, 0]
saturation_channel = hsv_spiderman_image[:, :, 1]
value_channel = hsv_spiderman_image[:, :, 2]

# Display the minimum and maximum values of each HSV plane
print(f"Min and Max of Hue Channel: {np.min(hue_channel)}, {np.max(hue_channel)}")
print(f"Min and Max of Saturation Channel: {np.min(saturation_channel)}, {np.max(saturation_channel)}")
print(f"Min and Max of Value Channel: {np.min(value_channel)}, {np.max(value_channel)}")

# Define the vibrancy transformation function
def vibrancy_transformation(input_value: int, alpha: float, sigma: int = 70) -> float:
    x = input_value
    return min(x + alpha * 128 * math.exp(-((x - 128) ** 2) / (2 * sigma ** 2)), 255)

# Initialize a dictionary to store the transformed images for different 'a' values
vibrancy_results = {0: None, 0.25: None, 0.5: None, 0.75: None, 1: None}

# Apply vibrancy transformation for different values of 'a' and display results
for alpha in vibrancy_results.keys():
    new_saturation = np.zeros(saturation_channel.shape, dtype=np.uint8)
    
    # Apply the vibrancy transformation to the saturation plane
    for i in range(saturation_channel.shape[0]):
        for j in range(saturation_channel.shape[1]):
            new_saturation[i, j] = vibrancy_transformation(saturation_channel[i, j], alpha)

    # Combine the transformed saturation with the original hue and value planes
    new_hsv_image = cv2.merge((hue_channel, new_saturation, value_channel))

    # Convert the HSV image back to BGR
    new_spiderman_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2BGR)

    # Store the result in the dictionary
    vibrancy_results[alpha] = new_spiderman_image

    # Plot the original and transformed images
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Display the original image
    axs[0].set_title("Original Image")
    axs[0].imshow(cv2.cvtColor(spiderman_image, cv2.COLOR_BGR2RGB))

    # Display the vibrancy-transformed image
    axs[1].set_title(f"Vibrancy Transformed Image (a={alpha})")
    axs[1].imshow(cv2.cvtColor(new_spiderman_image, cv2.COLOR_BGR2RGB))

    # Show the plot
    plt.show()

# Applying vibrancy transformation with alpha = 0.4
alpha = 0.4
new_saturation_plane = np.zeros(saturation_channel.shape, dtype=np.uint8)

# Apply vibrancy transformation to each pixel in the saturation plane
for i in range(saturation_channel.shape[0]):
    for j in range(saturation_channel.shape[1]):
        new_saturation_plane[i, j] = vibrancy_transformation(saturation_channel[i, j], alpha)

# Combine the transformed saturation plane with the original hue and value planes
new_hsv_image = cv2.merge((hue_channel, new_saturation_plane, value_channel))

# Convert back to BGR color space
final_spiderman_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2BGR)

# Plot the original and vibrancy-enhanced images side by side
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

# Display the original image
axs[0].set_title("Original Image")
axs[0].imshow(cv2.cvtColor(spiderman_image, cv2.COLOR_BGR2RGB))

# Display the vibrancy-enhanced image
axs[1].set_title(f"Vibrancy Enhanced Image (a={alpha})")
axs[1].imshow(cv2.cvtColor(final_spiderman_image, cv2.COLOR_BGR2RGB))

plt.show()

# Calculate and display the minimum and maximum saturation values
print(f"Min and Max Saturation: {saturation_channel.min()}, {saturation_channel.max()}")

# Generate the vibrancy transformation curve
alpha_value = 0.4
sigma_value = 70
input_pixel_values = np.arange(0, 256)
output_pixel_values = [vibrancy_transformation(x, alpha_value, sigma_value) for x in input_pixel_values]

# Plot the transformation curve
plt.figure(figsize=(8, 6))
plt.plot(input_pixel_values, output_pixel_values, label=f'Vibrancy Transformation (a={alpha_value}, σ={sigma_value})', color='red')
plt.xlabel('Input Pixel Value')
plt.ylabel('Output Pixel Value')
plt.title('Vibrancy Transformation Curve for Saturation Plane')
plt.legend()
plt.grid(True)
plt.show()
