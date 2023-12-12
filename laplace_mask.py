import cv2
import numpy as np
from scipy.signal import convolve2d 
def laplacian_filter(image, kernel_size=3):
    # Define a 3x3 Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convolve the image with the Laplacian kernel
    result = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')

    return result


def laplacian_sharpening(image, alpha=1.5, kernel_size=3):
    # Apply Laplacian filter
    laplacian = laplacian_filter(image, kernel_size)

    # Ensure the image is in the correct format
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply sharpening
    sharpened = image.astype(float) - alpha * laplacian

    # Clip values to be in the valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255)

    # Convert back to uint8
    sharpened = sharpened.astype(np.uint8)

    return sharpened

# Read the input image
input_image = cv2.imread('./x.png')

# Check if the image was loaded successfully
if input_image is None:
    print("Error: Unable to load the image.")
else:
    # Apply Laplacian sharpening
    sharpened_image = laplacian_sharpening(input_image)

    # Display the images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Laplacian Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
