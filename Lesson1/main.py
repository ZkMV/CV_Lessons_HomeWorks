import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=[10,5]

#source img 
#img = cv2.imread('C:\\Users\\mykhailo.zaika\\Documents\\PythonProjects\\CV_lessons\\Lesson1\\Homework_lesson1\\source.jpg')
img = cv2.imread('Lesson1/source.jpg')
plt.imshow(img)
plt.show() 
#---------------------------------------------
#Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.imshow(img)
plt.show()  
#---------------------------------------------
# Split the image into the three colour channels
red, green, blue = cv2.split(img)

# Compose the image in the RGB colour space
img1 = cv2.merge([red, green, blue])

# Split the image into the three colour channels
red, green, blue = cv2.split(img)

# Compose the image in the RGB colour space
img1 = cv2.merge([red, green, blue])

# Compose the image in the RBG colour space
img2 = cv2.merge([red, blue, green])

# Compose the image in the GRB colour space
img3 = cv2.merge([green, red, blue])

# Compose the image in the BGR colour space
img4 = cv2.merge([blue, red, green])

# Create the collage
out1 = np.hstack([img1, img2])
out2 = np.hstack([img3, img4])
out = np.vstack([out1, out2])

# Plot the collage
plt.imshow(out)
plt.axis(False)
plt.show()  
#---------------------------------------------
# Load an image (you can freely chose any image you like)
img = img = cv2.imread('C:\\Users\\mykhailo.zaika\\Documents\\PythonProjects\\CV_lessons\\Lesson1\\Homework_lesson1\\source.jpg')
plt.imshow(img)
# Convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Make a collage
img11 = img
img12 = cv2.flip(img11, 1)
img1_ = np.hstack([img11, img12])
img2_ = cv2.flip(img1_, 0)
img2 = np.vstack([img1_, img2_])

# Plot the collage
plt.imshow(img2)
plt.show()  
#---------------------------------------------
