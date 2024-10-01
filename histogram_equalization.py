import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_histogram_equalization(image):
    """
    Perform histogram equalization on a grayscale image.

    Args:
    image: Input grayscale image

    Returns:
    equalized_image: Image after histogram equalization
    """
    # Compute the histogram of the image
    histogram, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Compute the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    
    # Normalize the CDF to map the values between 0 and 255
    cdf_masked = np.ma.masked_equal(cdf, 0)  # Mask zeros to avoid division by zero
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf = np.ma.filled(cdf_normalized, 0).astype('uint8')  # Replace masked values with zeros
    
    # Use the CDF as a lookup table to transform the pixel values
    equalized_image = cdf[image]
    
    return equalized_image

def process_histogram_equalization(image_path):
    """
    Read the image, convert to grayscale, and apply histogram equalization.

    Args:
    image_path: Path to the input image file

    Returns:
    original_image: The original grayscale image
    equalized_image: The image after histogram equalization
    original_histogram: Histogram of the original image
    """
    # Load the image and convert it to grayscale
    original_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized_image = apply_histogram_equalization(grayscale_image)

    # Compute the histogram of the original image
    original_histogram, bins = np.histogram(grayscale_image.flatten(), 256, [0, 256])

    return grayscale_image, equalized_image, original_histogram

def display_results(original_image, equalized_image, original_histogram):
    """
    Display the original and equalized images alongside their histograms.

    Args:
    original_image: The original grayscale image
    equalized_image: The image after histogram equalization
    original_histogram: Histogram of the original image
    """
    # Compute the histogram of the equalized image
    equalized_histogram, bins = np.histogram(equalized_image.flatten(), 256, [0, 256])

    # Plot the original and equalized images along with their histograms
    plt.figure(figsize=(12, 8))

    # Display the original image
    plt.subplot(221)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Display the histogram of the original image
    plt.subplot(222)
    plt.plot(original_histogram, color='blue')
    plt.title('Original Image Histogram')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Display the equalized image
    plt.subplot(223)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    # Display the histogram of the equalized image
    plt.subplot(224)
    plt.plot(equalized_histogram, color='green')
    plt.title('Equalized Image Histogram')
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Load the image in grayscale
image_path = '1/Intensity-Transformations-and-Neighborhood-Filtering/a1images/shells.tif'

# Process the image to apply histogram equalization
original_image, equalized_image, original_histogram = process_histogram_equalization(image_path)

# Display the original and equalized images along with their histograms
display_results(original_image, equalized_image, original_histogram)
