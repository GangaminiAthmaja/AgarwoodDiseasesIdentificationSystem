import cv2
import numpy as np

# Read the image
image = cv2.imread('5.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold to separate the leaf from the background
_, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Create an output image with the same size and type as the original image
output = np.zeros_like(image)

# Use the binary mask to transfer the foreground (leaf) to the output image
output[binary_mask == 255] = image[binary_mask == 255]

# If you want the background to be white instead of black
output[binary_mask == 0] = [255, 255, 255]

# Save the resulting image
cv2.imwrite('leaf_no_background.jpg', output)

# If you want to view the image
cv2.imshow('Output Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()