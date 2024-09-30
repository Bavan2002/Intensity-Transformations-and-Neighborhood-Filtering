# # Gamma and Histograms
# import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np
# img_orig = cv.imread(r'C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\1\Intensity-Transformations-and-Neighborhood-Filtering\a1images\highlights_and_shadows.jpg', cv.IMREAD_COLOR)
# gamma = 1.5
# table = np.array([(i/255.0)**(1/gamma)*255.0 for i in np.arange(0,256)]).astype('uint8')
# img_gamma = cv.LUT(img_orig, table)
# img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
# img_gamma = cv.cvtColor(img_gamma, cv.COLOR_BGR2RGB)
# f, axarr = plt.subplots(3,2)
# axarr[0,0].imshow(img_orig)
# axarr[0,1].imshow(img_gamma)

# color = ('b', 'g', 'r')
# for i, c in enumerate(color):
#     hist_orig = cv.calcHist([img_orig], [i], None, [256], [0,256])
#     axarr[1,0].plot(hist_orig, color = c)
#     hist_gamma = cv.calcHist([img_gamma], [i], None, [256], [0,256])
#     axarr[1,1].plot(hist_gamma, color = c)    
# axarr[2,0].plot(table)
# axarr[2,0].set_xlim(0,255)
# axarr[2,0].set_ylim(0,255)
# axarr[2,0].set_aspect('equal')

# plt.show()

from skimage import color, exposure
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the image
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/highlights_and_shadows.jpg'  # Replace with the path to your image
image_rgb = np.array(Image.open(image_path))

# Convert the image from RGB to LAB color space
image_lab = color.rgb2lab(image_rgb)

# Separate the L, a, b channels
L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]

# Define gamma value for correction
gamma = 0.6  # You can modify this value based on your requirement

# Apply gamma correction to the L channel
L_gamma_corrected = exposure.adjust_gamma(L / 100.0, gamma) * 100  # Normalizing L to [0, 1] and applying gamma

# Merge the gamma-corrected L channel back with a and b
image_lab_corrected = np.stack((L_gamma_corrected, a, b), axis=2)

# Convert the corrected LAB image back to RGB color space
image_rgb_corrected = color.lab2rgb(image_lab_corrected)

# Plot the original and gamma-corrected images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(image_rgb_corrected)
ax[1].set_title('Gamma Corrected Image (γ = 0.5)')
ax[1].axis('off')

plt.show()

# Plot the histograms of the original and corrected L channel
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].hist(L.ravel(), bins=256, range=(0, 100), fc='black', ec='black')
ax[0].set_title('Histogram of Original L Channel')

ax[1].hist(L_gamma_corrected.ravel(), bins=256, range=(0, 100), fc='black', ec='black')
ax[1].set_title('Histogram of Gamma Corrected L Channel')

plt.show()


