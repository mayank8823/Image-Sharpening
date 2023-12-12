import cv2
import numpy as np

def gaussian_blur(image, kernel_size, sigma):
    # Create a 1D Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # Convolve the image with the 1D Gaussian kernel in both directions
    blurred_image = cv2.filter2D(image, -1, kernel)
    blurred_image = cv2.filter2D(blurred_image, -1, kernel.T)

    return blurred_image

def high_pass_filter(image, sigma=1):
    # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (low-pass filter)
    blurred_image = gaussian_blur(image, kernel_size=5, sigma=sigma)

    # Calculate the high-pass filter by subtracting the low-pass filtered image from the original image
    high_pass_image = image - blurred_image

    # Clip the values to the valid range [0, 255]
    high_pass_image = np.clip(high_pass_image, 0, 255).astype(np.uint8)

    return high_pass_image

# Read the input image
input_image = cv2.imread('./hp.png')
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

# Check if the image was loaded successfully
if input_image is None:
    print("Error: Unable to load the image.")
else:
    # Apply high-pass filter
    high_pass_image = high_pass_filter(input_image)

    # Display the images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('High-pass Filtered Image', high_pass_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

