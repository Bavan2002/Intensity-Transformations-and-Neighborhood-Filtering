# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt

# # c = np.array([(50,50),(50, 100), (150, 255), (150,150)])
# # t1 = np.linspace(0, c[0,1], c[0,0] + 1 - 0).astype('uint8')
# # print(len(t1))
# # t2 = np.linspace(c[0,1], c[1,1], c[1,0] - c[0,0]).astype('uint8')
# # print(len(t2))
# # t3 = np.linspace(c[1,1] + 1, c[2,1], c[2,0] - c[1,0]).astype('uint8') 
# # print(len(t3))
# # t4 = np.linspace(c[2,1], c[3,1], c[3,0] - c[2,0]).astype('uint8') 
# # print(len(t3))
# # t5 = np.linspace(c[3,1] + 1, 255, 255 - c[3,0]).astype('uint8') 
# # print(len(t3))
# # transform = np.concatenate((t1, t2), axis=0).astype('uint8')
# # transform = np.concatenate((transform, t3), axis=0).astype('uint8')
# # transform = np.concatenate((transform, t4), axis=0).astype('uint8')
# # transform = np.concatenate((transform, t5), axis=0).astype('uint8')

# transform = np.arange(255,-1, -1 ).astype('uint8')
# print(len(transform))

# plt.plot(transform)
# plt.xlabel('Input, $f(x)$')
# plt.ylabel('Output, $T[f(x)]$')
# plt.xlim(0,255)
# plt.ylim(0,255)
# # plt.grid(color='black', linestyle='-', linewidth=1)
# plt.savefig('transform.png')
# plt.show()

# img_orig = cv.imread(r'C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\1\Intensity-Transformations-and-Neighborhood-Filtering\a1images\brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)
# cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
# cv.imshow("Image", img_orig)
# cv.waitKey(0)
# image_transformed = cv.LUT(img_orig, transform)
# cv.imshow("Image", image_transformed)
# cv.waitKey(0)
# cv.destroyAllWindows()


# transform = np.arange(0, 256, 1 ).astype('uint8')
# print(len(transform))

# plt.plot(transform)
# plt.xlabel('Input, $f(x)$')
# plt.ylabel('Output, $T[f(x)]$')
# plt.xlim(0,255)
# plt.ylim(0,255)
# plt.savefig('transform.png')
# plt.show()

# img_orig = cv.imread(r'C:\Users\Bavan2002.DESKTOP-TITUVCT\Desktop\EN3160_Assignment\1\Intensity-Transformations-and-Neighborhood-Filtering\a1images\brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)
# cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
# cv.imshow("Image", img_orig)
# cv.waitKey(0)
# image_transformed = cv.LUT(img_orig, transform)
# cv.imshow("Image", image_transformed)
# cv.waitKey(0)
# cv.destroyAllWindows()


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice.png'  # Replace with your image path
image = Image.open(image_path).convert('L')  # Convert image to grayscale

# Convert image to numpy array
image_array = np.array(image)

# Display the original image
plt.figure(figsize=(6, 6))
plt.imshow(image_array, cmap='gray')
plt.title('Original Brain Proton Density Image')
plt.axis('off')
plt.show()

# Get the intensity values and plot the histogram for analysis
plt.figure(figsize=(6, 4))
plt.hist(image_array.ravel(), bins=256, range=(0, 256), fc='black', ec='black')
plt.title('Intensity Histogram of the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Define thresholds for white and gray matter based on intensity histogram
# These ranges are assumptions based on typical brain imaging patterns
white_matter_lower = 180
white_matter_upper = 230

gray_matter_lower = 100
gray_matter_upper = 180

# Create masks to highlight white and gray matter separately
white_matter_mask = np.logical_and(image_array >= white_matter_lower, image_array <= white_matter_upper)
gray_matter_mask = np.logical_and(image_array >= gray_matter_lower, image_array <= gray_matter_upper)

# Create images for white matter and gray matter
white_matter_image = np.zeros_like(image_array)
gray_matter_image = np.zeros_like(image_array)

white_matter_image[white_matter_mask] = image_array[white_matter_mask]
gray_matter_image[gray_matter_mask] = image_array[gray_matter_mask]

# Display the accentuated white matter and gray matter images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(white_matter_image, cmap='gray')
ax[0].set_title('Accentuated White Matter')
ax[0].axis('off')

ax[1].imshow(gray_matter_image, cmap='gray')
ax[1].set_title('Accentuated Gray Matter')
ax[1].axis('off')

plt.show()
