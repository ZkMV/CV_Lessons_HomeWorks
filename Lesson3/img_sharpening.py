import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]


img = cv2.imread('Lesson3/source/kodim05.jpg')
if img is None:
    raise FileNotFoundError("Зображення не знайдено за вказаним шляхом!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show() 

# Create a blurred (unsharp) version of the original image (you can use Gaussian blurring)
unsharp = cv2.GaussianBlur(img, (9, 9), 10.0)
plt.imshow(unsharp)
plt.show() 

# Create the difference image (original − unsharp)
# Note: Remember that you are working with uint8 data types. Any addition or substractions
# might result in overflow or underflow, respectively. You can prevent this by casting the images to float.
#diff = cv2.subtract(img.astype(np.float32), unsharp.astype(np.float32))
diff = img.astype(np.float32) - unsharp.astype(np.float32)
#plt.imshow(diff + 128)  # Normalize for visualization - не працює
plt.imshow(diff / 255.0 + 0.5)  # Normalize for visualization
plt.show()

# Apply USM to get the resulting image using `sharpened = original + (original − unsharp) × amount`
# Note: Again, take care of underflows/overflows if necessary.
sharpened = img.astype(np.float32) + diff * 1.5
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)  # Clip values to valid range and convert back to uint8
plt.imshow(sharpened)
plt.show()