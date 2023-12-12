import cv2
import numpy as np


def nlm_sharpening(image, h=10, templateWindowSize=7, searchWindowSize=21):
    # Convert the image to grayscale if it is a color image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Apply the NLM filter for sharpening
    sharpened = cv2.fastNlMeansDenoising(gray_image, None, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

    # Calculate the sharpened image by subtracting the denoised image from the original
    sharpened = cv2.addWeighted(gray_image, 1.5, sharpened, -0.5, 0)

    # Clip values to be in the valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


def non_local_means(image, h=10, template_size=(5, 5), search_size=(11, 11)):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply non-local means filtering
    denoised_image = cv2.fastNlMeansDenoising(image, None, h, template_size[0], search_size[0])

    return denoised_image

# Read the input image
image = cv2.imread('./harry.png')
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Check if the image was loaded successfully
if input_image is None:
    print("Error: Unable to load the image.")
else:
    # Apply non-local means filtering
    result_image = nlm_sharpening(input_image)

    # Display the images
    cv2.imshow('Original Image', input_image)
    cv2.imshow('output Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
