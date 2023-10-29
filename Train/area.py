import cv2
import numpy as np
from rembg import remove
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def area_cal(image_path):
    # Read the image
    image = cv2.imread(image_path)

    output_image = remove(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to create a mask of the leaf
    _, leaf_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use the mask to create an image with only the leaf and no background
    leaf_only = cv2.bitwise_and(image, image, mask=leaf_mask)

    # Now, work with the 'leaf_only' image for further processing
    gray_leaf_only = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2GRAY)
    blurred_leaf_only = cv2.GaussianBlur(gray_leaf_only, (5, 5), 0)

    # Threshold to separate the diseased parts from the healthy parts of the leaf
    _, disease_thresh = cv2.threshold(blurred_leaf_only, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate the areas
    total_leaf_area = np.sum(leaf_mask == 255)
    disease_area = np.sum(disease_thresh == 255)
    percentage_diseased = (disease_area / total_leaf_area) * 100

    print(f"Total Leaf Area: {total_leaf_area} pixels")
    print(f"Diseased Area: {disease_area} pixels")
    print(f"Percentage of Diseased Area: {percentage_diseased:.2f}%")

    return total_leaf_area, disease_area, percentage_diseased

