import numpy as np
from PIL import Image

# Create a 2x2 RGB image (red and blue pixels) 2x2x3 array
rgb_data = np.array([[[255, 0, 0], [0, 0, 255]], 
                     [[0, 255, 0], [255, 255, 0]]], dtype=np.uint8)

pil_img = Image.fromarray(rgb_data, mode='RGB')
print(pil_img)