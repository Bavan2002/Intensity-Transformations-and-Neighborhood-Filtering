import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image for gamma correction
bgr_image = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/highlights_and_shadows.jpg')

# Convert the image from BGR to LAB color space
lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

# Split the LAB image into L, A, and B channels
L_channel, A_channel, B_channel = cv2.split(lab_image)

# Normalize the L channel to the range [0, 1] for gamma correction
L_normalized = L_channel / 255.0

# Apply gamma correction with gamma = 2.2
gamma_high = 2
L_gamma_high = np.power(L_normalized, gamma_high)

# Rescale the L channel back to the range [0, 255] and convert to uint8
L_gamma_high = np.uint8(L_gamma_high * 255)

# Merge the gamma corrected L channel with the original A and B channels
lab_gamma_corrected_high = cv2.merge((L_gamma_high, A_channel, B_channel))

# Convert back to the BGR color space
gamma_corrected_high_img = cv2.cvtColor(lab_gamma_corrected_high, cv2.COLOR_LAB2BGR)

# Apply gamma correction with gamma = 0.5
gamma_low = 0.5
L_gamma_low = np.power(L_normalized, gamma_low)

# Rescale and convert to uint8
L_gamma_low = np.uint8(L_gamma_low * 255)

# Merge the gamma corrected L channel with the original A and B channels
lab_gamma_corrected_low = cv2.merge((L_gamma_low, A_channel, B_channel))

# Convert back to BGR color space
gamma_corrected_low_img = cv2.cvtColor(lab_gamma_corrected_low, cv2.COLOR_LAB2BGR)

# Display the original and gamma corrected images side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Display gamma correction with gamma = 0.5
axes[0].set_title("Gamma Corrected (γ = 0.5)")
axes[0].imshow(cv2.cvtColor(gamma_corrected_low_img, cv2.COLOR_BGR2RGB))

# Display the original image
axes[1].set_title("Original Image")
axes[1].imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))

# Display gamma correction with gamma = 2
axes[2].set_title("Gamma Corrected (γ = 2)")
axes[2].imshow(cv2.cvtColor(gamma_corrected_high_img, cv2.COLOR_BGR2RGB))

# Show the plots
plt.show()

# Histogram plotting for original and gamma corrected images
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Histogram for the original image (RGB channels)
axs[0].set_title("Original Image Histogram (RGB Channels)")
axs[0].hist(bgr_image[:, :, 0].flatten(), bins=256, range=[0, 256], color='b', alpha=0.4, label='Blue Channel')
axs[0].hist(bgr_image[:, :, 1].flatten(), bins=256, range=[0, 256], color='g', alpha=0.4, label='Green Channel')
axs[0].hist(bgr_image[:, :, 2].flatten(), bins=256, range=[0, 256], color='r', alpha=0.4, label='Red Channel')
axs[0].set_xlim([0, 256])
axs[0].legend()

# Histogram for gamma corrected image with gamma = 2.2
axs[1].set_title("Gamma Corrected (γ = 2) Histogram")
axs[1].hist(gamma_corrected_high_img[:, :, 0].flatten(), bins=256, range=[0, 256], color='b', alpha=0.4, label='Blue Channel')
axs[1].hist(gamma_corrected_high_img[:, :, 1].flatten(), bins=256, range=[0, 256], color='g', alpha=0.4, label='Green Channel')
axs[1].hist(gamma_corrected_high_img[:, :, 2].flatten(), bins=256, range=[0, 256], color='r', alpha=0.4, label='Red Channel')
axs[1].set_xlim([0, 256])
axs[1].set_ylim([0, 8000])
axs[1].legend()

plt.tight_layout()
plt.show()

# Plot histograms for the gamma corrected image with gamma = 0.5
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Histogram for the original image
axs[0].set_title("Original Image Histogram (RGB Channels)")
axs[0].hist(bgr_image[:, :, 0].flatten(), bins=256, range=[0, 256], color='b', alpha=0.4, label='Blue Channel')
axs[0].hist(bgr_image[:, :, 1].flatten(), bins=256, range=[0, 256], color='g', alpha=0.4, label='Green Channel')
axs[0].hist(bgr_image[:, :, 2].flatten(), bins=256, range=[0, 256], color='r', alpha=0.4, label='Red Channel')
axs[0].set_xlim([0, 256])
axs[0].legend()

# Histogram for the gamma corrected image with gamma = 0.5
axs[1].set_title("Gamma Corrected (γ = 0.5) Histogram")
axs[1].hist(gamma_corrected_low_img[:, :, 0].flatten(), bins=256, range=[0, 256], color='b', alpha=0.4, label='Blue Channel')
axs[1].hist(gamma_corrected_low_img[:, :, 1].flatten(), bins=256, range=[0, 256], color='g', alpha=0.4, label='Green Channel')
axs[1].hist(gamma_corrected_low_img[:, :, 2].flatten(), bins=256, range=[0, 256], color='r', alpha=0.4, label='Red Channel')
axs[1].set_xlim([0, 256])
axs[1].set_ylim([0, 8000])
axs[1].legend()

# Show the figure
plt.tight_layout()
plt.show()
