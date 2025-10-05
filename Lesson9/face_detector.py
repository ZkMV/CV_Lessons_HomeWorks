import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import os
plt.rcParams['figure.figsize'] = [15, 10]


casc_path = 'Lesson9/source1/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(casc_path)

img = cv2.imread('Lesson9/source1/faces.jpg')
if img is None:
    raise FileNotFoundError("Зображення не знайдено за вказаним шляхом!")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)
plt.show()



'''
Viola-Jones
Viola-Jones is a classic and powerful algorithm for object detection.
It is a sliding window approach that work in cascades and exploits Haar transform (basis functions) to learn object descriptors.
It also makes uso of boosting.
'''
# minNeighbors = 0 shows all the detection at all scale, a value of approx. 5 shall felter out all the spurious detections
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

print('Number of detected faces:', len(faces))

# Draw rectangle around each face
result = np.copy(img)
faces_img = []
for (x, y, w, h) in faces: 
    # Draw rectangle around the face
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    faces_img.append(img[y:y+h, x:x+w, :])
    
plt.imshow(result)
plt.show()

plt.subplot(221), plt.imshow(result, cmap='gray')
plt.subplot(222), plt.imshow(faces_img[0])
plt.subplot(223), plt.imshow(faces_img[1])
#plt.subplot(224), plt.imshow(faces_img[2])
plt.show()

'''
Face Detection via dlib
Dlib is a general purpose cross-platform software library that contains many useful tools. In particular, it includes a trained DNN for face detection.
'''
import dlib
# Let's load the detector
detector = dlib.get_frontal_face_detector()
# Detect faces, see http://dlib.net/face_detector.py.html
# 1 --> upsampling factor
rects = detector(gray, 1)

print('Number of detected faces:', len(rects))
print(rects)
print(rects[0].left)

def rect_to_bb(rect):
    # Dlib rect --> OpenCV rect
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


# Draw rectangle around each face
result_dlib = np.copy(img)
faces_dlib_img = []
for rect in rects:    
    # Draw rectangle around the face
    x, y, w, h = rect_to_bb(rect)
    print(x, y, w, h)
    cv2.rectangle(result_dlib, (x, y), (x+w, y+h), (0, 255, 0), 3)
    faces_dlib_img.append(img[y:y+h, x:x+w, :])
    

plt.subplot(121), plt.imshow(result), plt.title('Viola-Jones')
plt.subplot(122), plt.imshow(result_dlib), plt.title('dlib')
plt.show()


#Завантажуємо зображення обличча, яке будемо шукати серед детектованих
face_1 = cv2.imread('Lesson9/source1/face1.jpg')
if face_1 is None:
    raise FileNotFoundError("Зображення не знайдено за вказаним шляхом!")   
face_1 = cv2.cvtColor(face_1, cv2.COLOR_BGR2RGB)
plt.imshow(face_1)  
plt.show()
face_1 = cv2.cvtColor(face_1, cv2.COLOR_RGB2GRAY)
face_1 = cv2.resize(face_1, (100, 100)) 
plt.imshow(face_1, cmap='gray')  
plt.show()
#Обчислюємо дескриптор обличчя, яке будемо шукати
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(face_1, None)
print('Face 1 - keypoints:', len(kp1), 'descriptors:', des1.shape)
img_kp1 = cv2.drawKeypoints(face_1, kp1, None)
plt.imshow(img_kp1, cmap='gray')
plt.show()

#Тепер обчислюємо дескриптори для кожного з детектованих обличь
sift = cv2.SIFT_create()
faces_kp = []
faces_des = []
for i, face in enumerate(faces_dlib_img):
    face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face_gray = cv2.resize(face_gray, (100, 100)) 
    kp, des = sift.detectAndCompute(face_gray, None)
    faces_kp.append(kp)
    faces_des.append(des)
    print('Face', i, '- keypoints:', len(kp), 'descriptors:', des.shape)
    img_kp = cv2.drawKeypoints(face_gray, kp, None)
    plt.subplot(1, len(faces_dlib_img), i+1), plt.imshow(img_kp, cmap='gray') 
plt.show()
#Порівнюємо дескриптори обличчя, яке шукаємо, з дескрипторами детектованих обличь
bf = cv2.BFMatcher()  
matches_all = []
for i, des2 in enumerate(faces_des):
    matches = bf.knnMatch(des1, des2, k=2) 
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    matches_all.append(good)
    print('Face', i, '- matches:', len(matches), 'good matches:', len(good))
    img_matches = cv2.drawMatchesKnn(face_1, kp1, faces_dlib_img[i], faces_kp[i], good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.subplot(1, len(faces_dlib_img), i+1), plt.imshow(img_matches)
plt.show()
#Обираємо обличчя з найбільшою кількістю співпадінь
best_face_idx = np.argmax([len(m) for m in matches_all])    
print('Best face index:', best_face_idx)
plt.imshow(faces_dlib_img[best_face_idx])
plt.show()   
#Виводимо детектоване обличчя на оригінальному зображенні
(x, y, w, h) = rect_to_bb(rects[best_face_idx])
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
plt.imshow(img) 
plt.show()

