import cv2
import numpy as np



def histogram_equalization(image):
    # Flatten the image to a 1D array
    flattened_image = image.flatten()

    # Calculate the histogram of the flattened image
    histogram, bins = np.histogram(flattened_image, bins=256, range=[0, 256])

    # Calculate the cumulative distribution function (CDF) of the histogram
    cdf = histogram.cumsum()

    # Normalize the CDF to be in the range [0, 255]
    cdf_normalized = (cdf * 255) / cdf[-1]

    # Use the normalized CDF values as the new intensity values
    equalized_image = np.interp(flattened_image, bins[:-1], cdf_normalized)

    # Reshape the 1D array back to the original shape of the image
    equalized_image = equalized_image.reshape(image.shape)

    # Convert to unsigned 8-bit integer (if necessary)
    equalized_image = equalized_image.astype(np.uint8)

    return equalized_image

# Read the input image
input_image = cv2.imread('./nature.png',cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if input_image is None:
    print("Error: Unable to load the image.")
else:
    # Apply histogram equalization
    equalized_image = histogram_equalization(input_image)

    # Display the images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
