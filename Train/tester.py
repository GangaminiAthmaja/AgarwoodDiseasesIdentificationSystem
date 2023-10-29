import cv2
import numpy as np
from rembg import remove
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def area_cal(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image = remove(image)
    # Display the original image using matplotlib (optional)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis
    plt.show()

    # Perform background removal
    output_image = remove(image)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV range for the color of the disease
    lower_color = np.array([hue_min, saturation_min, value_min])
    upper_color = np.array([hue_max, saturation_max, value_max])

    # Create a mask to identify the disease color within the specified range
    disease_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Apply Gaussian blur to the mask (optional)
    disease_mask = cv2.GaussianBlur(disease_mask, (5, 5), 0)

    # Calculate the areas
    total_leaf_area = np.sum(disease_mask == 255)
    percentage_diseased = (total_leaf_area / (disease_mask.shape[0] * disease_mask.shape[1])) * 100

    print(f"Total Leaf Area: {total_leaf_area} pixels")
    print(f"Percentage of Diseased Area: {percentage_diseased:.2f}%")

    return total_leaf_area, percentage_diseased

# Define the HSV range for the color of the disease (you need to determine these values)
hue_min = 0
hue_max = 30
saturation_min = 50
saturation_max = 255
value_min = 50
value_max = 255

area_cal('2.jpg')
