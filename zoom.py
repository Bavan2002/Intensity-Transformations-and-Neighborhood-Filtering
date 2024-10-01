# Import required libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to zoom the image
def zoom_image(image, scale, interpolation):
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=interpolation)

# Function to compute normalized SSD between two images
def compute_normalized_ssd(img1, img2, bypass_size_error=True):
    if not bypass_size_error:
        # Ensure images are of the same size
        assert img1.shape == img2.shape, "Images must be the same shape for SSD computation."
    else: 
        # Crop the larger image to match the smaller image
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])

        img1 = img1[:min_height, :min_width]
        img2 = img2[:min_height, :min_width]

    
    
    # Compute the sum of squared differences
    ssd = np.sum((img1.astype("float32") - img2.astype("float32")) ** 2)
    
    # Normalize by the number of pixels
    norm_ssd = ssd / np.prod(img1.shape)
    
    return norm_ssd

# Function to display images side-by-side
def display_images(original, nearest, bilinear, titles):
    plt.figure(figsize=(15, 15))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(titles[0])
    plt.axis('off')

    # Nearest-neighbor zoomed image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(nearest, cv2.COLOR_BGR2RGB))
    plt.title(titles[1])
    plt.axis('off')

    # Bilinear zoomed image
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(bilinear, cv2.COLOR_BGR2RGB))
    plt.title(titles[2])
    plt.axis('off')

    plt.show()


def get_zoom_and_orignal_img(small_img, big_img, scale_factor=4, bypass_size_error=True):
        # Scale factor
    scale_factor = 4

    # Zoom using nearest-neighbor interpolation
    zoomed_nn = zoom_image(small_img, scale_factor, cv2.INTER_NEAREST)

    # Zoom using bilinear interpolation
    zoomed_bilinear = zoom_image(small_img, scale_factor, cv2.INTER_LINEAR)

    # Compute normalized SSD for nearest-neighbor zoomed image
    ssd_nn = compute_normalized_ssd(big_img, zoomed_nn, bypass_size_error=bypass_size_error)

    # Compute normalized SSD for bilinear zoomed image
    ssd_bilinear = compute_normalized_ssd(big_img, zoomed_bilinear, bypass_size_error=bypass_size_error)

    # Print out SSD values
    print(f"Normalized SSD (Nearest Neighbor): {ssd_nn}")
    print(f"Normalized SSD (Bilinear): {ssd_bilinear}")

    # Display the images
    titles = ["Original Image", "Nearest Neighbor Zoomed", "Bilinear Zoomed"]
    display_images(big_img, zoomed_nn, zoomed_bilinear, titles)
    

im01_small_img = cv2.imread('C:/Users/Bavan2002.DESKTOP-TITUVCT/Desktop/EN3160_Assignment/1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03small.png')
im01_img = cv2.imread('C:/Users/Bavan2002.DESKTOP-TITUVCT/Desktop/EN3160_Assignment/1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03.png')

# Assuming 'im01_small_img' is your image
height, width = im01_small_img.shape[:2]

# Scaling the dimensions by a factor of 4
new_height = height * 4
new_width = width * 4

# Resize the image using the new dimensions
resized_image = cv2.resize(im01_small_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Display the resized image

# im02_small_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im02small.png')
# im02_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im02.png')

# im03_small_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03small.png')
# im03_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/im03.png')

# taylor_very_small_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/taylor_very_small.jpg')
# taylor_small_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/taylor_small.jpg')
# taylor_img = cv2.imread('1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/a1q5images/=taylor.jpg')

get_zoom_and_orignal_img(small_img=resized_image, big_img=im01_img, scale_factor=4, bypass_size_error=False)
