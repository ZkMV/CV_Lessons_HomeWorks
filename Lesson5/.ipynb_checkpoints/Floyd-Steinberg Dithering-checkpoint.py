import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import os
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('Lesson5/source/kodim23.png')
if img is None:
    raise FileNotFoundError("Зображення не знайдено за вказаним шляхом!")

#load and show the camera frame.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
plt.imshow(img)
plt.show() 

# Black, dark gray, light gray, white
colors = np.array([[0, 0, 0],
                   [64, 64, 64],
                   [192, 192, 192],
                   [255, 255, 255]])

# Cast the image to float
img = img.astype(float)

# Prepare for quantization
rows, cols, channels = img.shape
quantized = np.zeros_like(img)

# Apply quantization
for r in range(rows):
    for c in range(cols):
        # Extract the original pixel value
        pixel = img[r, c]
        
        # Find the closest colour from the pallette (using absolute value/Euclidean distance)
        # Note: You may need more than one line of code here
        new_pixel = colors[np.argmin(np.sum((colors - pixel) ** 2, axis=1))]
        
        # Apply quantization
        quantized[r, c, :] = new_pixel
        
# Show quantized image (don't forget to cast back to uint8)
plt.imshow(quantized.astype(np.uint8))
plt.show() 

# Compute average quantization error
avg_quant_error = np.mean(np.abs(img - quantized))
print(f"Average quantization error: {avg_quant_error}")


# Make a temporal copy of the original image, we will need it for error diffusion
img_tmp = np.copy(img)
dithering = np.zeros_like(img)

for r in range(1, rows-1):
    for c in range(1, cols-1):
        pixel = img_tmp[r, c]
                # Find the closest colour from the pallette (using absolute value/Euclidean distance)
        # Note: You may need more than one line of code here
        new_pixel = colors[np.argmin(np.sum((colors - pixel) ** 2, axis=1))]
        
                # Compute quantization error
        quant_error = pixel - new_pixel
        
        # Diffuse the quantization error accroding to the FS diffusion matrix
        # Note: You may need more than one line of code here
        img_tmp = img_tmp.astype(float)
        
        # Apply dithering
        dithering[r, c, :] = new_pixel

# Show quantized image (don't forget to cast back to uint8)
# optimally quantized
plt.subplot(121), plt.imshow(quantized.astype(np.uint8)), plt.title('Quantized')
# dithering
plt.subplot(122), plt.imshow(dithering.astype(np.uint8)), plt.title('Dithering')
plt.show()

# Compute average quantization error for dithered image
avg_dith_error = np.mean(np.abs(img - dithering))
print(f"Average dithering error: {avg_dith_error}")

