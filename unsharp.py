import cv2
import numpy as np
def gaussian_blur(image, kernel_size, sigma):
    # Create a 1D Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # Convolve the image with the 1D Gaussian kernel in both directions
    blurred_image = cv2.filter2D(image, -1, kernel)
    blurred_image = cv2.filter2D(blurred_image, -1, kernel.T)

    return blurred_image

def unsharp_masking(image, sigma=1.5, strength=1.5):
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Calculate the sharpened image by subtracting the blurred image from the original
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    # Clip values to be in the valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened
# Read the input image
input_image = cv2.imread('./x.png')

# Check if the image was loaded successfully
if input_image is None:
    print("Error: Unable to load the image.")
else:
    # Apply unsharp masking
    sharpened_image = unsharp_masking(input_image)

    # Display the images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
