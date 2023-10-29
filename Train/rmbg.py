from rembg import remove
from PIL import Image
import cv2

input_path = "2.jpg"
output_path = "output2.png"

input_image = Image.open(input_path)
output_image = remove(input_image)
output_image.save(output_path)