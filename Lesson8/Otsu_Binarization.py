# Lesson8/Otsu_Binarization.py

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import os
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('Lesson8/source/document.jpg')
if img is None:
    raise FileNotFoundError("Зображення не знайдено за вказаним шляхом!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Let's plot the image
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(122), plt.imshow(gray, cmap='gray')
plt.show()

#First, let's have a look at the histogram.
h = np.histogram(gray.ravel(), bins=256, range=(0, 256))
plt.bar(h[1][0:-1], h[0])
plt.xlabel('Colour'), plt.ylabel('Count')
plt.grid(True)
plt.show()

'''
Otsu Thresholding
Let's now implement the Otsu thresholding algorithm. 
Remember that the algorithm consists of an optimization process that finds the thresholds 
that minimizes the intra-class variance or, equivalently, maximizes the inter-class variance.

In this homework, you are going to demonstrate the working principle of the Otsu algorithm. 
Therefore, you won't have to worry about an efficient implementation, 
we are going to use the brute force approach here.
'''

# Get image dimensions
rows, cols = gray.shape
# Compute the total amount of image pixels
num_pixel = rows * cols

# Initializations
best_wcv = 1e6  # Best within-class variance (wcv)
opt_th = None   # Threshold corresponding to the best wcv

# Brute force search using all possible thresholds (levels of gray)
for th in range(0, 256):
    # Extract the image pixels corresponding to the foreground
    foreground = gray[gray >= th]
    # Extract the image pixels corresponding to the background
    background = gray[gray < th]
    
    # If foreground or background are empty, continue
    if len(foreground) == 0 or len(background) == 0:
        continue
    
    # Compute class-weights (omega parameters) for foreground and background
    omega_f = len(foreground) / num_pixel
    omega_b = len(background) / num_pixel
    
    # Compute pixel variance for foreground and background (var function from numpy)
    # https://numpy.org/doc/stable/reference/generated/numpy.var.html
    sigma2_f = np.var(foreground)
    sigma2_b = np.var(background)
    
    # Compute the within-class variance
    wcv = omega_f * sigma2_f + omega_b * sigma2_b
    
    # Perform the optimization
    if wcv < best_wcv:
        best_wcv = wcv
        opt_th = th
        
# Print out the optimal threshold found by Otsu algorithm
print('Optimal threshold', opt_th)

#Finally, let's compare the original image and its thresholded representation.

# 1. Застосовуємо поріг до зображення у відтінках сірого ('gray')
# 2. Перетворюємо булевий масив (True/False) у числовий (1/0) типу uint8
binary_image = ((gray > opt_th).astype(np.uint8)) * 255  # 0/255 для коректного відображення/збереження

# --- Report (4 panels) у простому форматі subplot(221..224) ---
out_path = 'Lesson8/result/otsu_result.png'

plt.figure(figsize=(12, 9))

plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.subplot(122), plt.imshow(gray, cmap='gray'); plt.axis('off')
plt.show()

plt.subplot(223)
plt.bar(h[1][0:-1], h[0])
plt.axvline(opt_th, linestyle='--')  # вертикальна лінія порога
plt.xlabel('Colour')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(224), plt.imshow(binary_image, cmap='gray'); plt.axis('off')
plt.title(f'Thresholded Image (th={opt_th})')

# Save the result
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")

plt.show()

'''
- Looking at the computed histogram, could it be considered bimodal?
    я би не назвав її (гістограму) бімодальною, оскільки є більше двох піків, хоча третій пік має невелику площу.
- Looking at the computed histogram, what binarization threshold would you chose? Why?
    я би обрав поріг біля 150, примруживши праве око, але, якщо серйозно, то виглядає, 
    що біля 150 проходдить водорозділ між двома основними "скупченнями" навколо основних піків.
- Looking at the resulting (thresholded) image, is the text binarization (detection) good?
    ні, дуже відносно. Текст на передьному плані після бінаризації читається гарно, але далі робиться менш читабельним. 
    При тому, щона оригінальі, я можу розрізнити текст на задньому плані    
'''
