import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import imageio
import os
from screeninfo import get_monitors
from PIL import Image, ImageDraw
#import torch
from collections import deque

# Центральна константа для збереження результатів
OUTPUT_FOLDER = os.path.join('Lesson10', 'result')

#video1 = cv2.VideoCapture('Lesson10/source/birds_in_sky.mp4')
#video1 = cv2.VideoCapture('source//hawk.mp6')
output_folder = OUTPUT_FOLDER

#VIDEO = cv2.VideoCapture('Lesson10/source/birds_in_sky.mp4')
VIDEO = cv2.VideoCapture('Lesson10/source/hawk5.mp4')
#--------------------------------Функції--------------------------------#
# Функція для створення GIF з набору зображень (застосовуємо далі для створення GIF з результатами)
def create_gif(images, output_filename, fps=10):
    """
    Creates a GIF from a list of images.
    
    Args:
        images (list): A list of images (NumPy arrays from OpenCV).
        output_filename (str): The path and name for the output GIF file.
        fps (int): Frames per second for the GIF.
    """
    # Перевірки і нормалізація входут
    if not images:
        raise ValueError("No images provided to create_gif")

    # Видалити можливі None-фрейми
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        raise ValueError("No valid (non-None) frames to write to GIF")

    # Забезпечити однаковий розмір кадрів і правильний тип
    h, w = valid_images[0].shape[:2]
    norm_images = []
    for idx, img in enumerate(valid_images):
        if img is None:
            continue
        # Якщо розміри відрізняються, змінити їх до першого
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        # Переконатися, що це uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        # Конвертувати BGR -> RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        norm_images.append(rgb)

    if not norm_images:
        raise ValueError("After normalization there are no frames to save")

    # Переконатися, що директорія для збереження існує
    out_dir = os.path.dirname(output_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Спробувати зберегти GIF і надати більш інформативний вивід у випадку помилки
    try:
        imageio.mimsave(output_filename, norm_images, fps=fps)
        print(f"Successfully saved GIF to {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Failed to save GIF to {output_filename}: {e}")
        raise

#------------------------------------------------------------------------#
#------------------------------Part 3: KCF--------------------------------#
#------------------------------------------------------------------------#
'''
print("\nStarting Part 3: KCF Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_kcf = VIDEO
# Використовуємо константу OUTPUT_FOLDER для збереження всіх результатів
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Створюємо об'єкт трекера KCF
tracker = cv2.TrackerKCF_create()

# 3. Зчитуємо перший кадр
ret, first_frame = video_kcf.read()
if not ret:
    print("Failed to read video")
    exit()

# 4. Дозволяємо користувачу вибрати об'єкт для відстеження
print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI("Select Bird to Track", first_frame, False)
cv2.destroyWindow("Select Bird to Track")

# 5. Ініціалізуємо трекер першим кадром та вибраною рамкою
tracker.init(first_frame, bbox)

# 6. Готуємося до циклу
gif_frames_kcf = []

while True:
    ret, frame = video_kcf.read()
    if not ret:
        print("End of video for KCF tracking.")
        break

    # 7. Оновлюємо трекер
    ok, bbox = tracker.update(frame)

    # 8. Малюємо рамку
    if ok:
        # Відстеження успішне
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 6, 1)
        cv2.putText(frame, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Відстеження не вдалося
        cv2.putText(frame, "Tracking failure detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Додаємо кадр для GIF
    gif_frames_kcf.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow("KCF Tracker", frame)

    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Зберігаємо результат у GIF
if gif_frames_kcf:
    gif_filename_kcf = 'kcf_tracking.gif'
    output_path_kcf = os.path.join(output_folder, gif_filename_kcf)
    
    print(f"Creating KCF tracking GIF with {len(gif_frames_kcf)} frames...")
    create_gif(gif_frames_kcf, output_path_kcf, fps=10) # Збільшимо fps для плавності

# 10. Звільняємо ресурси
video_kcf.release()
cv2.destroyAllWindows()
print("KCF tracking finished.")
'''
'''
#------------------------------------------------------------------------#
#-----------------------------Part 4: CSRT-------------------------------#
#------------------------------------------------------------------------#
print("\nStarting Part 4: CSRT Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# --- CSRT CONFIGURATION PARAMETERS ---
# Set these parameters to experiment with the tracker's behaviour:
# 1. Use HOG features (shape/gradients). Try setting to False if colour is more distinct than shape.
USE_HOG_FEATURES = True 

# 2. Number of bins for the color histogram. Higher values (e.g., 32 or 64) increase colour
#    discrimination, which is useful when the background is "uniform" green. Default is 16.
HISTOGRAM_BINS = 64 # Try 16, 32, or 64

# 1. Initialisation
video_csrt = VIDEO
# Use the OUTPUT_FOLDER constant to save all results
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Create the CSRT Tracker object with custom parameters
# a) Get default parameters
params = cv2.TrackerCSRT_Params()

# b) Apply custom settings
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS

# c) Create the tracker
tracker = cv2.TrackerCSRT_create(params)

# 3. Read the first frame
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

# 4. Allow the user to select the object for tracking
print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI("Select Bird to Track", first_frame, False)
cv2.destroyWindow("Select Bird to Track")

# 5. Initialize the tracker with the first frame and the selected box
tracker.init(first_frame, bbox)

# 6. Prepare for the loop
gif_frames_csrt = []
while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of video for CSRT tracking.")
        break
    
    # 7. Update the tracker
    ok, bbox = tracker.update(frame)
    
    # 8. Draw the bounding box
    if ok:
        # Tracking successful
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 6, 1)
        cv2.putText(frame, "Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    # Add frame for GIF
    gif_frames_csrt.append(frame)
    
    # Show the real-time result (optional)
    cv2.imshow("CSRT Tracker", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 9. Save the result to GIF
if gif_frames_csrt:
    gif_filename_csrt = 'csrt_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating CSRT tracking GIF with {len(gif_frames_csrt)} frames...")
    # NOTE: 'create_gif' function is assumed to be defined elsewhere
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) 

# 10. Release resources
video_csrt.release()
cv2.destroyAllWindows()
print("CSRT tracking finished.")
'''


'''
#------------------------------------------------------------------------#
#-----------------Part 4: Hybrid CSRT + Template Matching----------------#
#------------------------------------------------------------------------#

# --- CONFIGURATION PARAMETERS ---
# CSRT parameters 
USE_HOG_FEATURES = True 
HISTOGRAM_BINS = 32 # Try 16, 32, or 64

# Template Matching parameters
# Template matching method: cv2.TM_CCOEFF_NORMED is generally the most robust
TEMPLATE_METHOD = cv2.TM_CCOEFF_NORMED 
SEARCH_WINDOW_SIZE = 150 # Pixels around the last known position to search (e.g., 150x150 pixels)
TEMPLATE_THRESHOLD = 0.6 # Minimum correlation value to accept a template match (0.0 to 1.0)



print("\nStarting Part 4: Hybrid CSRT + Template Matching Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Створюємо об'єкт трекера CSRT з параметрами
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

# 4. Дозволяємо користувачу вибрати об'єкт для відстеження
print("Please select a bounding box for the bird and press ENTER or SPACE.")
# Get the initial bounding box
bbox = cv2.selectROI("Select Bird to Track", first_frame, False)
cv2.destroyWindow("Select Bird to Track")

# Convert bbox to integers
x, y, w, h = [int(v) for v in bbox]

# --- Template Matching Initialisation ---
# Extract the initial bird image (template)
template = first_frame[y:y+h, x:x+w]

# Check if the template is valid (i.e., not empty)
if template.size == 0:
    print("Error: The selected region is empty. Exiting.")
    exit()

# 5. Ініціалізуємо CSRT
tracker.init(first_frame, bbox)

# 6. Готуємося до циклу
gif_frames_csrt = []
current_bbox = bbox # Store the last known good bounding box

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of video for hybrid tracking.")
        break

    # 7. Оновлюємо CSRT
    ok, new_bbox_csrt = tracker.update(frame)

    # 8. Логіка відновлення: Template Matching
    if ok:
        current_bbox = new_bbox_csrt # CSRT succeeded, update the last known good box
    else:
        # CSRT failed, attempt recovery using Template Matching
        
        # Define the search area (ROI) around the last known good position
        cx, cy = int(current_bbox[0] + current_bbox[2]/2), int(current_bbox[1] + current_bbox[3]/2)
        
        # Define ROI coordinates, ensuring they stay within the frame boundaries
        roi_x_start = max(0, cx - SEARCH_WINDOW_SIZE)
        roi_y_start = max(0, cy - SEARCH_WINDOW_SIZE)
        roi_x_end = min(frame.shape[1], cx + SEARCH_WINDOW_SIZE)
        roi_y_end = min(frame.shape[0], cy + SEARCH_WINDOW_SIZE)
        
        # Get the search area
        search_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        if search_roi.size > 0:
            # Perform Template Matching
            res = cv2.matchTemplate(search_roi, template, TEMPLATE_METHOD)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # Check if the match is good enough
            if max_val >= TEMPLATE_THRESHOLD:
                # Match found! Calculate the new bounding box coordinates
                top_left_x = max_loc[0] + roi_x_start
                top_left_y = max_loc[1] + roi_y_start
                
                bbox_matched = (top_left_x, top_left_y, w, h)
                
                # Re-initialise the CSRT tracker with the new position
                tracker.init(frame, bbox_matched)
                current_bbox = bbox_matched # Update the last known good box
                ok = True # Mark as successful after re-initialization
                cv2.putText(frame, "Template Restored", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2) # Orange colour
            else:
                # No good match found in the search area
                cv2.putText(frame, f"Template match failed (Val:{max_val:.2f})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
    # Set the bbox variable to the latest good box for drawing
    bbox_to_draw = current_bbox if ok else current_bbox 
    
    # 9. Малюємо рамку
    if ok:
        # Tracking successful (either by CSRT or Template Restored)
        p1 = (int(bbox_to_draw[0]), int(bbox_to_draw[1]))
        p2 = (int(bbox_to_draw[0] + bbox_to_draw[2]), int(bbox_to_draw[1] + bbox_to_draw[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 6, 1)
        cv2.putText(frame, "CSRT Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Tracking failure detected
        cv2.putText(frame, "Tracking failure detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # We still draw the last known good box, but in red, to show where it was lost
        p1 = (int(current_bbox[0]), int(current_bbox[1]))
        p2 = (int(current_bbox[0] + current_bbox[2]), int(current_bbox[1] + current_bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 6, 1) # Red box
        
    # Додаємо кадр для GIF
    gif_frames_csrt.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow("Hybrid Tracker (CSRT+Template)", frame)
    
    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 10. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_template_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating Hybrid Template tracking GIF with {len(gif_frames_csrt)} frames...")
    # NOTE: 'create_gif' function is assumed to be defined elsewhere
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) 

# 11. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("Hybrid tracking finished.")
'''


'''
#------------------------------------------------------------------------#
#---------------------Part 5: Hybrid CSRT + Optical Flow-----------------#
#------------------------------------------------------------------------#

# --- CONFIGURATION PARAMETERS ---
# CSRT parameters (as discussed earlier)
USE_HOG_FEATURES = True 
HISTOGRAM_BINS = 32 # Try 16, 32, or 64

# Optical Flow parameters
MAX_CORNERS = 100 # Maximum number of points (corners) to track
FEATURE_QUALITY = 0.3 # Minimum accepted quality of image corners
MIN_DISTANCE = 7 # Minimum distance between points

# Define the criteria for the optical flow search
LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


print("\nStarting Part 4: Hybrid CSRT + Optical Flow Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Створюємо об'єкт трекера CSRT з параметрами
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

# Convert the first frame to grayscale for feature detection
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 4. Дозволяємо користувачу вибрати об'єкт для відстеження
print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI("Select Bird to Track", first_frame, False)
cv2.destroyWindow("Select Bird to Track")

# 5. Ініціалізуємо CSRT
tracker.init(first_frame, bbox)

# --- Optical Flow Initialisation ---
x, y, w, h = [int(v) for v in bbox]
bbox_mask = np.zeros_like(old_gray)
bbox_mask[y:y+h, x:x+w] = 255 # Create a mask over the selected area

# Find initial key points (corners) within the selected bounding box
old_points = cv2.goodFeaturesToTrack(old_gray, mask=bbox_mask, 
                                     maxCorners=MAX_CORNERS, 
                                     qualityLevel=FEATURE_QUALITY, 
                                     minDistance=MIN_DISTANCE)
# Ensure we have points to track
if old_points is None or len(old_points) < 5:
    print("Not enough distinctive features detected for Optical Flow.")

# 6. Готуємося до циклу
gif_frames_csrt = []

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of video for hybrid tracking.")
        break

    # Convert the new frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 7. Оновлюємо CSRT
    ok, bbox = tracker.update(frame)
    
    # 8. Оновлюємо Optical Flow (як механізм відновлення)
    if old_points is not None:
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **LK_PARAMS)

        # Select only the good points (status == 1)
        good_new = new_points[status == 1]
        good_old = old_points[status == 1]

        # Calculate the median movement offset from the key points
        if len(good_new) > 5: # Require a minimum number of good points
            # Calculate the median displacement
            dx = np.median(good_new[:, 0] - good_old[:, 0])
            dy = np.median(good_new[:, 1] - good_old[:, 1])

            # Apply median displacement to the CSRT bounding box (for potential re-initialization)
            bbox_flow = (bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3])
            
            # --- Restoration/Re-initialization Logic ---
            # If CSRT failed, use Optical Flow to estimate new position and re-init
            if not ok:
                # Check if the flow movement is reasonable (e.g., within frame)
                if bbox_flow[0] > 0 and bbox_flow[1] > 0 and bbox_flow[0] + bbox_flow[2] < frame.shape[1] and bbox_flow[1] + bbox_flow[3] < frame.shape[0]:
                    tracker.init(frame, bbox_flow) # Re-initialise CSRT
                    bbox = bbox_flow # Update bbox with the flow estimate
                    ok = True # Mark as successful after re-initialization
                    cv2.putText(frame, "Flow Restored", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2) # Orange colour
            
            # Update points for the next iteration
            old_gray = frame_gray.copy()
            old_points = good_new.reshape(-1, 1, 2)
        else:
            # If too few points are tracked, re-detect them in the current BBOX
            # This helps if points were lost due to blur or rotation
            x, y, w, h = [int(v) for v in bbox]
            bbox_mask = np.zeros_like(frame_gray)
            bbox_mask[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)] = 255
            
            new_points_re = cv2.goodFeaturesToTrack(frame_gray, mask=bbox_mask, 
                                                maxCorners=MAX_CORNERS, 
                                                qualityLevel=FEATURE_QUALITY, 
                                                minDistance=MIN_DISTANCE)
            old_points = new_points_re # Use re-detected points for the next frame
    
    # 9. Малюємо рамку
    if ok:
        # Tracking successful (either by CSRT or Flow Restored)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 6, 1)
        cv2.putText(frame, "CSRT Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Tracking failure detected
        cv2.putText(frame, "Tracking failure detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
    # Додаємо кадр для GIF
    gif_frames_csrt.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow("Hybrid Tracker (CSRT+Flow)", frame)
    
    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 10. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_optical_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating Hybrid tracking GIF with {len(gif_frames_csrt)} frames...")
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) # Assuming create_gif exists

# 11. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("Hybrid tracking finished.")
'''

'''
#------------------------------------------------------------------------#
#-----------------Part 5: Hybrid CSRT + Template + Kalman----------------#
#------------------------------------------------------------------------#
import cv2
import numpy as np
import os 
# Assuming you have the 'create_gif' function available

# --- CONFIGURATION PARAMETERS ---
# CSRT parameters 
USE_HOG_FEATURES = True 
HISTOGRAM_BINS = 32

# Template Matching parameters
TEMPLATE_METHOD = cv2.TM_CCOEFF_NORMED 
SEARCH_WINDOW_SIZE = 150 
TEMPLATE_THRESHOLD = 0.6 

# Kalman Filter parameters
# We track 4 state variables: [x, y, vx, vy] - centre position and velocity
STATE_SIZE = 4
# We measure 2 variables: [x, y] - centre position from the tracker
MEASUREMENT_SIZE = 2
# We assume no control input (no acceleration input from outside)
CONTROL_SIZE = 0 


print("\nStarting Part 4: Hybrid CSRT + Template + Kalman Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Створюємо об'єкт трекера CSRT з параметрами
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

# 4. Дозволяємо користувачу вибрати об'єкт для відстеження
print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI("Select Bird to Track", first_frame, False)
cv2.destroyWindow("Select Bird to Track")

# Convert bbox to integers and extract template
x, y, w, h = [int(v) for v in bbox]
template = first_frame[y:y+h, x:x+w]
if template.size == 0:
    print("Error: The selected region is empty. Exiting.")
    exit()

# 5. Ініціалізуємо CSRT
tracker.init(first_frame, bbox)

# --- Kalman Filter Initialisation ---
kf = cv2.KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE, CONTROL_SIZE)
# Transition Matrix (A): Predicts next state based on current state (x = x + vx, y = y + vy)
kf.transitionMatrix = np.array([[1, 0, 1, 0],   # x = x + vx
                                [0, 1, 0, 1],   # y = y + vy
                                [0, 0, 1, 0],   # vx = vx
                                [0, 0, 0, 1]], np.float32) # vy = vy

# Measurement Matrix (H): Relates state vector to measurement vector (we only measure x, y)
kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                 [0, 1, 0, 0]], np.float32)

# Process Noise Covariance (Q): How much the system state might change (e.g., due to acceleration)
# Higher Q means the filter trusts its model less and relies more on new measurements.
kf.processNoiseCov = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], np.float32) * 0.05

# Measurement Noise Covariance (R): How much we trust the measurement from the tracker (CSRT/Template)
# Higher R means the filter trusts the tracker less.
kf.measurementNoiseCov = np.array([[1, 0], 
                                   [0, 1]], np.float32) * 2

# Set initial state (initial centre position and zero velocity)
initial_cx, initial_cy = x + w/2, y + h/2
kf.statePost = np.array([[initial_cx], [initial_cy], [0.], [0.]], np.float32)

# Measurement vector
measurement = np.array((2, 1), np.float32) 

# 6. Готуємося до циклу
gif_frames_csrt = []
current_bbox = bbox # Last known good bounding box

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of hybrid tracking.")
        break
        
    # --- 7. KALMAN FILTER PREDICTION ---
    # Predict the next state (center x, y)
    kf_prediction = kf.predict()
    predicted_cx, predicted_cy = kf_prediction[0][0], kf_prediction[1][0]
    
    # Calculate the predicted BBOX based on the predicted centre and the original size
    bbox_predicted = (predicted_cx - w/2, predicted_cy - h/2, w, h)

    # 8. Оновлюємо CSRT
    ok, new_bbox_csrt = tracker.update(frame)

    # 9. Логіка відновлення: Template Matching (centred on Kalman Prediction)
    success_flag = ok
    
    if not ok:
        # CSRT failed, attempt recovery using Template Matching centered on the prediction
        
        # We centre the search area (ROI) around the predicted position
        cx, cy = int(predicted_cx), int(predicted_cy)
        
        # Define ROI coordinates, ensuring they stay within the frame boundaries
        roi_x_start = max(0, cx - SEARCH_WINDOW_SIZE)
        roi_y_start = max(0, cy - SEARCH_WINDOW_SIZE)
        roi_x_end = min(frame.shape[1], cx + SEARCH_WINDOW_SIZE)
        roi_y_end = min(frame.shape[0], cy + SEARCH_WINDOW_SIZE)
        
        # Get the search area
        search_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        if search_roi.size > 0:
            # Perform Template Matching
            res = cv2.matchTemplate(search_roi, template, TEMPLATE_METHOD)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= TEMPLATE_THRESHOLD:
                # Match found! Calculate the new bounding box coordinates
                top_left_x = max_loc[0] + roi_x_start
                top_left_y = max_loc[1] + roi_y_start
                
                bbox_matched = (top_left_x, top_left_y, w, h)
                
                # Re-initialise the CSRT tracker with the new position
                tracker.init(frame, bbox_matched)
                new_bbox_csrt = bbox_matched # Use the matched box for correction
                success_flag = True # Mark as successful after re-initialization
                cv2.putText(frame, "Template Restored", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2)
            else:
                # Template match failed
                cv2.putText(frame, f"Match Failed (Val:{max_val:.2f})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
    # --- 10. KALMAN FILTER CORRECTION ---
    if success_flag:
        # Get the measured center from the successful tracker update/restoration
        bx, by, bw, bh = [float(v) for v in new_bbox_csrt]
        measured_cx, measured_cy = bx + bw/2, by + bh/2
        
        # Correct the Kalman Filter state
        measurement[0] = measured_cx
        measurement[1] = measured_cy
        kf_corrected = kf.correct(measurement)
        
        # Use the corrected position for the next display and the next prediction
        corrected_cx, corrected_cy = kf_corrected[0][0], kf_corrected[1][0]
        
        # Update current_bbox using the Kalman-corrected centre
        current_bbox = (corrected_cx - w/2, corrected_cy - h/2, w, h)
        bbox_to_draw = current_bbox
        
    else:
        # If both CSRT and Template Matching failed, use the PREDICTED position from the Kalman Filter
        # This keeps the tracking running based on inertia until a new measurement is found.
        bbox_to_draw = bbox_predicted 
        current_bbox = bbox_to_draw # Update the last known good box to the prediction
        cv2.putText(frame, "Kalman Prediction", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2) # Magenta colour
        
    # 11. Малюємо рамку
    # Draw based on the final determined position (corrected or predicted)
    p1 = (int(bbox_to_draw[0]), int(bbox_to_draw[1]))
    p2 = (int(bbox_to_draw[0] + bbox_to_draw[2]), int(bbox_to_draw[1] + bbox_to_draw[3]))
    color = (0, 255, 0) if success_flag else (0, 0, 255) # Green if successful, Red if only prediction
    
    cv2.rectangle(frame, p1, p2, color, 6, 1)
    cv2.putText(frame, "Tracking Success" if success_flag else "Tracking Failure", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Додаємо кадр для GIF
    gif_frames_csrt.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow("Hybrid Tracker (CSRT+Template+Kalman)", frame)
    
    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 12. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_kalman_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating Hybrid Kalman tracking GIF with {len(gif_frames_csrt)} frames...")
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) 

# 13. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("Hybrid tracking finished.")
'''


'''
#------------------------------------------------------------------------#
#----------Part 5: HYBRID CSRT + Optical Flow + Kalman + Haar------------#
#------------------------------------------------------------------------#

# --- CONFIGURATION PARAMETERS ---
# CSRT parameters 
USE_HOG_FEATURES = True 
HISTOGRAM_BINS = 32

# Optical Flow parameters
MAX_CORNERS = 100
FEATURE_QUALITY = 0.3
MIN_DISTANCE = 7
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Kalman Filter parameters
STATE_SIZE = 4
MEASUREMENT_SIZE = 2
KF_PROCESS_NOISE_COV = 0.05 
KF_MEASUREMENT_NOISE_COV = 2 

# Haar Cascade Recovery parameters (USING THE /source FOLDER)
# The path is set to look inside the 'source' subdirectory
HAAR_CASCADE_FILENAME = "source/bird-cascade.xml" 
HAAR_SCALE_FACTOR = 1.4
HAAR_MIN_NEIGHBOURS = 5
HAAR_MAX_SIZE = (30, 30) 
HAAR_SEARCH_RADIUS = 50 


print("\nStarting Part 4: HYBRID CSRT + Optical Flow + Kalman + Haar Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація та Завантаження Детектора
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# Load the Haar Cascade Detector
# Now uses the path 'source/bird-cascade.xml'
bird_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILENAME)
if bird_cascade.empty():
    print(f"ERROR: Haar Cascade file '{HAAR_CASCADE_FILENAME}' not loaded. Cannot use detection for recovery.")
    bird_cascade = None

# 2. Створюємо об'єкт трекера CSRT
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 4. Дозволяємо користувачу вибрати об'єкт
print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI("Select Bird to Track", first_frame, False)
cv2.destroyWindow("Select Bird to Track")

x, y, w, h = [int(v) for v in bbox]

# 5. Ініціалізуємо CSRT, Optical Flow та Kalman Filter (КОД БЕЗ ЗМІН)
tracker.init(first_frame, bbox)

bbox_mask = np.zeros_like(old_gray)
mask_x, mask_y, mask_w, mask_h = max(0, x - 10), max(0, y - 10), w + 20, h + 20
bbox_mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 255 
old_points = cv2.goodFeaturesToTrack(old_gray, mask=bbox_mask, maxCorners=MAX_CORNERS, qualityLevel=FEATURE_QUALITY, minDistance=MIN_DISTANCE)
last_good_points = old_points.copy() if old_points is not None else None

kf = cv2.KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(STATE_SIZE, dtype=np.float32) * KF_PROCESS_NOISE_COV
kf.measurementNoiseCov = np.eye(MEASUREMENT_SIZE, dtype=np.float32) * KF_MEASUREMENT_NOISE_COV

initial_cx, initial_cy = x + w/2, y + h/2
kf.statePost = np.array([[initial_cx], [initial_cy], [0.], [0.]], np.float32)
measurement = np.zeros((MEASUREMENT_SIZE, 1), np.float32) 

# 6. Готуємося до циклу
gif_frames_csrt = []
current_bbox = bbox 
last_successful_bbox = bbox

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of hybrid tracking.")
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # 7. KALMAN FILTER PREDICTION
    kf_prediction = kf.predict()
    predicted_cx, predicted_cy = kf_prediction[0][0], kf_prediction[1][0]
    
    # 8. Оновлюємо CSRT
    ok, new_bbox_csrt = tracker.update(frame)
    success_flag = ok
    recovery_source = None
    
    # --- 9. RECOVERY LOGIC (Flow -> Haar Cascade) ---
    
    if not ok:
        
        # A) First attempt recovery with Optical Flow (local movement verification)
        if last_good_points is not None and len(last_good_points) >= 5:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, last_good_points, None, **LK_PARAMS)
            good_new = new_points[status == 1]
            good_old = last_good_points[status == 1]
            
            if len(good_new) >= 5:
                dx = np.median(good_new[:, 0] - good_old[:, 0])
                dy = np.median(good_new[:, 1] - good_old[:, 1])
                
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    flow_corrected_cx = predicted_cx + dx
                    flow_corrected_cy = predicted_cy + dy
                    new_bbox_csrt = (flow_corrected_cx - w/2, flow_corrected_cy - h/2, w, h)
                    
                    if 0 < flow_corrected_cx < frame.shape[1] and 0 < flow_corrected_cy < frame.shape[0]:
                        tracker.init(frame, new_bbox_csrt)
                        success_flag = True
                        recovery_source = "Flow"
        
        # B) Second attempt recovery with Haar Cascade (object detection)
        if not success_flag and bird_cascade is not None:
            
            min_size = (w // 2, h // 2)
            
            search_x1 = int(max(0, predicted_cx - HAAR_SEARCH_RADIUS))
            search_y1 = int(max(0, predicted_cy - HAAR_SEARCH_RADIUS))
            search_x2 = int(min(frame.shape[1], predicted_cx + HAAR_SEARCH_RADIUS))
            search_y2 = int(min(frame.shape[0], predicted_cy + HAAR_SEARCH_RADIUS))

            search_roi_gray = frame_gray[search_y1:search_y2, search_x1:search_x2]
            
            if search_roi_gray.size > 0:
                detections = bird_cascade.detectMultiScale(
                    search_roi_gray, 
                    scaleFactor=HAAR_SCALE_FACTOR, 
                    minNeighbors=HAAR_MIN_NEIGHBOURS, 
                    maxSize=HAAR_MAX_SIZE, 
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(detections) > 0:
                    (dx, dy, dw, dh) = detections[0]
                    
                    detected_x = dx + search_x1
                    detected_y = dy + search_y1
                    detected_w, detected_h = dw, dh
                    
                    new_bbox_csrt = (detected_x, detected_y, detected_w, detected_h)
                    tracker.init(frame, new_bbox_csrt)
                    success_flag = True
                    recovery_source = "Haar"

    # --- 10. KALMAN FILTER CORRECTION ---
    if success_flag:
        # 1. Update Optical Flow points
        bx, by, bw, bh = [int(v) for v in new_bbox_csrt]
        mask_c = np.zeros_like(frame_gray)
        mask_c[max(0, by):min(frame.shape[0], by+bh), max(0, bx):min(frame.shape[1], bx+bw)] = 255
        
        points_re = cv2.goodFeaturesToTrack(frame_gray, mask=mask_c, maxCorners=MAX_CORNERS, qualityLevel=FEATURE_QUALITY, minDistance=MIN_DISTANCE)
        last_good_points = points_re if points_re is not None else last_good_points
        
        # 2. Kalman Correction
        bx, by, bw, bh = [float(v) for v in new_bbox_csrt]
        measured_cx, measured_cy = bx + bw/2, by + bh/2
        
        measurement[0] = measured_cx
        measurement[1] = measured_cy
        kf_corrected = kf.correct(measurement)
        
        corrected_cx, corrected_cy = kf_corrected[0][0], kf_corrected[1][0]
        current_bbox = (corrected_cx - w/2, corrected_cy - h/2, w, h) 
        bbox_to_draw = current_bbox
        
        if recovery_source:
            cv2.putText(frame, f"{recovery_source} Restored & Init", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2)
        else:
            cv2.putText(frame, "CSRT Tracking", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
    else:
        # If all methods failed, rely purely on Kalman Prediction
        predicted_w, predicted_h = w, h
        current_bbox = (predicted_cx - predicted_w/2, predicted_cy - predicted_h/2, predicted_w, predicted_h)
        bbox_to_draw = current_bbox
        cv2.putText(frame, "Kalman Prediction Only", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        
    old_gray = frame_gray.copy() 

    # 11. Малюємо рамку та виводимо FPS
    p1 = (int(bbox_to_draw[0]), int(bbox_to_draw[1]))
    p2 = (int(bbox_to_draw[0] + bbox_to_draw[2]), int(bbox_to_draw[1] + bbox_to_draw[3]))
    color = (0, 255, 0) if success_flag else (0, 0, 255) 
    
    cv2.rectangle(frame, p1, p2, color, 6, 1)
    cv2.putText(frame, "Tracking Success" if success_flag else "Tracking Failure", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Додаємо кадр для GIF
    gif_frames_csrt.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow("Hybrid Tracker (CSRT+Flow+Kalman+Haar)", frame)
    
    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 12. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_flow_kalman_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating Hybrid Haar tracking GIF with {len(gif_frames_csrt)} frames...")
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) 

# 13. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("Hybrid tracking finished.")
'''

'''
#------------------------------------------------------------------------#
#----------Part 5: HYBRID CSRT + Optical Flow + Kalman + Haar------------#
#------------------------------------------------------------------------#
try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except ImportError:
    print("Install 'screeninfo' (pip install screeninfo) for accurate window centering.")
    SCREENINFO_AVAILABLE = False


# --- Helper Function for Window Centering ---
def center_window(window_name, frame_width, frame_height):
    """Calculates coordinates to center the window on the primary monitor."""
    if not SCREENINFO_AVAILABLE:
        # Use a hardcoded, safe fallback position if screeninfo is not available
        cv2.moveWindow(window_name, 100, 50)
        return

    try:
        # Get primary monitor info
        monitor = get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height
        
        # Calculate new X and Y coordinates
        x = (screen_width - frame_width) // 2
        y = (screen_height - frame_height) // 2 - 20 # Slight offset up for title bar
        
        # Ensure coordinates are not negative
        x = max(0, x)
        y = max(0, y)
        
        cv2.moveWindow(window_name, x, y)
        
    except Exception as e:
        print(f"Warning: Centering failed: {e}. Using default position.")
        cv2.moveWindow(window_name, 100, 50) 

# --- CONFIGURATION PARAMETERS ---
# CSRT parameters 
USE_HOG_FEATURES = True 
HISTOGRAM_BINS = 32

# Optical Flow parameters
MAX_CORNERS = 100
FEATURE_QUALITY = 0.3
MIN_DISTANCE = 7
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Kalman Filter parameters
STATE_SIZE = 4
MEASUREMENT_SIZE = 2
KF_PROCESS_NOISE_COV = 0.05 
KF_MEASUREMENT_NOISE_COV = 2 

# Haar Cascade Recovery parameters 
HAAR_CASCADE_FILENAME = "source/bird-cascade.xml" 
HAAR_SCALE_FACTOR = 1.4
HAAR_MIN_NEIGHBOURS = 5
HAAR_MAX_SIZE = (30, 30) 
HAAR_SEARCH_RADIUS = 50 


print("\nStarting Part 4: HYBRID CSRT + Optical Flow + Kalman + Haar Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація та Завантаження Детектора
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

bird_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILENAME)
if bird_cascade.empty():
    print(f"ERROR: Haar Cascade file '{HAAR_CASCADE_FILENAME}' not loaded. Cannot use detection for recovery.")
    bird_cascade = None

# 2. Створюємо об'єкт трекера CSRT
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
FRAME_H, FRAME_W, _ = first_frame.shape # <- GET FRAME SIZE

# 4. Дозволяємо користувачу вибрати об'єкт
WINDOW_ROI = "Select Bird to Track"
cv2.namedWindow(WINDOW_ROI)
center_window(WINDOW_ROI, FRAME_W, FRAME_H) # <- CENTER THE ROI WINDOW

print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)
cv2.destroyWindow(WINDOW_ROI)

x, y, w, h = [int(v) for v in bbox]

# 5. Ініціалізуємо CSRT, Optical Flow та Kalman Filter 
tracker.init(first_frame, bbox)

bbox_mask = np.zeros_like(old_gray)
mask_x, mask_y, mask_w, mask_h = max(0, x - 10), max(0, y - 10), w + 20, h + 20
bbox_mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 255 
old_points = cv2.goodFeaturesToTrack(old_gray, mask=bbox_mask, maxCorners=MAX_CORNERS, qualityLevel=FEATURE_QUALITY, minDistance=MIN_DISTANCE)
last_good_points = old_points.copy() if old_points is not None else None

kf = cv2.KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(STATE_SIZE, dtype=np.float32) * KF_PROCESS_NOISE_COV
kf.measurementNoiseCov = np.eye(MEASUREMENT_SIZE, dtype=np.float32) * KF_MEASUREMENT_NOISE_COV

initial_cx, initial_cy = x + w/2, y + h/2
kf.statePost = np.array([[initial_cx], [initial_cy], [0.], [0.]], np.float32)
measurement = np.zeros((MEASUREMENT_SIZE, 1), np.float32) 

# 6. Готуємося до циклу
gif_frames_csrt = []
current_bbox = bbox 
last_successful_bbox = bbox
WINDOW_TRACKER = "Hybrid Tracker (CSRT+Flow+Kalman+Haar)"

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of hybrid tracking.")
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # 7. KALMAN FILTER PREDICTION
    kf_prediction = kf.predict()
    predicted_cx, predicted_cy = kf_prediction[0][0], kf_prediction[1][0]
    
    # 8. Оновлюємо CSRT
    ok, new_bbox_csrt = tracker.update(frame)
    success_flag = ok
    recovery_source = None
    
    # --- 9. RECOVERY LOGIC (Flow -> Haar Cascade) ---
    if not ok:
        
        # A) First attempt recovery with Optical Flow (local movement verification)
        if last_good_points is not None and len(last_good_points) >= 5:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, last_good_points, None, **LK_PARAMS)
            good_new = new_points[status == 1]
            good_old = last_good_points[status == 1]
            
            if len(good_new) >= 5:
                dx = np.median(good_new[:, 0] - good_old[:, 0])
                dy = np.median(good_new[:, 1] - good_old[:, 1])
                
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    flow_corrected_cx = predicted_cx + dx
                    flow_corrected_cy = predicted_cy + dy
                    new_bbox_csrt = (flow_corrected_cx - w/2, flow_corrected_cy - h/2, w, h)
                    
                    if 0 < flow_corrected_cx < frame.shape[1] and 0 < flow_corrected_cy < frame.shape[0]:
                        tracker.init(frame, new_bbox_csrt)
                        success_flag = True
                        recovery_source = "Flow"
        
        # B) Second attempt recovery with Haar Cascade (object detection)
        if not success_flag and bird_cascade is not None:
            
            min_size = (w // 2, h // 2)
            
            search_x1 = int(max(0, predicted_cx - HAAR_SEARCH_RADIUS))
            search_y1 = int(max(0, predicted_cy - HAAR_SEARCH_RADIUS))
            search_x2 = int(min(frame.shape[1], predicted_cx + HAAR_SEARCH_RADIUS))
            search_y2 = int(min(frame.shape[0], predicted_cy + HAAR_SEARCH_RADIUS))

            search_roi_gray = frame_gray[search_y1:search_y2, search_x1:search_x2]
            
            if search_roi_gray.size > 0:
                detections = bird_cascade.detectMultiScale(
                    search_roi_gray, 
                    scaleFactor=HAAR_SCALE_FACTOR, 
                    minNeighbors=HAAR_MIN_NEIGHBOURS, 
                    maxSize=HAAR_MAX_SIZE, 
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(detections) > 0:
                    (dx, dy, dw, dh) = detections[0]
                    
                    detected_x = dx + search_x1
                    detected_y = dy + search_y1
                    detected_w, detected_h = dw, dh
                    
                    new_bbox_csrt = (detected_x, detected_y, detected_w, detected_h)
                    tracker.init(frame, new_bbox_csrt)
                    success_flag = True
                    recovery_source = "Haar"

    # --- 10. KALMAN FILTER CORRECTION ---
    if success_flag:
        # 1. Update Optical Flow points
        bx, by, bw, bh = [int(v) for v in new_bbox_csrt]
        mask_c = np.zeros_like(frame_gray)
        mask_c[max(0, by):min(frame.shape[0], by+bh), max(0, bx):min(frame.shape[1], bx+bw)] = 255
        
        points_re = cv2.goodFeaturesToTrack(frame_gray, mask=mask_c, maxCorners=MAX_CORNERS, qualityLevel=FEATURE_QUALITY, minDistance=MIN_DISTANCE)
        last_good_points = points_re if points_re is not None else last_good_points
        
        # 2. Kalman Correction
        bx, by, bw, bh = [float(v) for v in new_bbox_csrt]
        measured_cx, measured_cy = bx + bw/2, by + bh/2
        
        measurement[0] = measured_cx
        measurement[1] = measured_cy
        kf_corrected = kf.correct(measurement)
        
        corrected_cx, corrected_cy = kf_corrected[0][0], kf_corrected[1][0]
        current_bbox = (corrected_cx - w/2, corrected_cy - h/2, w, h) 
        bbox_to_draw = current_bbox
        
        if recovery_source:
            cv2.putText(frame, f"{recovery_source} Restored & Init", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2)
        else:
            cv2.putText(frame, "CSRT Tracking", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
    else:
        # If all methods failed, rely purely on Kalman Prediction
        predicted_w, predicted_h = w, h
        current_bbox = (predicted_cx - predicted_w/2, predicted_cy - predicted_h/2, predicted_w, predicted_h)
        bbox_to_draw = current_bbox
        cv2.putText(frame, "Kalman Prediction Only", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        
    old_gray = frame_gray.copy() 

    # 11. Малюємо рамку та виводимо FPS
    p1 = (int(bbox_to_draw[0]), int(bbox_to_draw[1]))
    p2 = (int(bbox_to_draw[0] + bbox_to_draw[2]), int(bbox_to_draw[1] + bbox_to_draw[3]))
    color = (0, 255, 0) if success_flag else (0, 0, 255) 
    
    cv2.rectangle(frame, p1, p2, color, 6, 1)
    cv2.putText(frame, "Tracking Success" if success_flag else "Tracking Failure", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Додаємо кадр для GIF
    gif_frames_csrt.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow(WINDOW_TRACKER, frame)
    
    # --- CENTER THE TRACKING WINDOW ONCE ---
    if len(gif_frames_csrt) == 1:
        center_window(WINDOW_TRACKER, FRAME_W, FRAME_H) # <- CENTER THE TRACKING WINDOW
    
    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 12. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_haar_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating Hybrid Haar tracking GIF with {len(gif_frames_csrt)} frames...")
    # NOTE: 'create_gif' function is assumed to be defined elsewhere
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) 

# 13. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("Hybrid tracking finished.")
'''

'''
#------------------------------------------------------------------------#
#----------Part X: Hybrid CSRT + Kalman (Motion Compensation)------------#
#------------------------------------------------------------------------#
# --- Global Window Position ---
WINDOW_START_X = 50
WINDOW_START_Y = 50

# --- CONFIGURATION PARAMETERS ---
# CSRT parameters 
USE_HOG_FEATURES = True 
HISTOGRAM_BINS = 32

# Kalman Filter parameters
# We track 4 state variables: [x, y, vx, vy] - RELATIVE centre position and RELATIVE velocity
STATE_SIZE = 4
MEASUREMENT_SIZE = 2
KF_PROCESS_NOISE_COV = 0.05  # Trust the model more (smoother tracking)
KF_MEASUREMENT_NOISE_COV = 2  # Trust the CSRT measurement less (smoother prediction)

# Global Flow parameters (for motion compensation)
# We use a simple goodFeaturesToTrack on the whole frame to estimate global motion
FLOW_MAX_CORNERS = 500
FLOW_QUALITY = 0.01 
FLOW_MIN_DISTANCE = 30
FLOW_LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# --- Helper Function for Window Position ---
# Defined without screeninfo, using fixed position for unification
def set_window_position(window_name, x, y):
    cv2.moveWindow(window_name, x, y)



print("\nStarting Part X: Hybrid CSRT + Kalman (Motion Compensation) Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Створюємо об'єкт трекера CSRT
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
FRAME_H, FRAME_W, _ = first_frame.shape

# 4. Дозволяємо користувачу вибрати об'єкт
WINDOW_ROI = "Select Bird to Track"
cv2.namedWindow(WINDOW_ROI)
set_window_position(WINDOW_ROI, WINDOW_START_X, WINDOW_START_Y)

print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)
cv2.destroyWindow(WINDOW_ROI)

x, y, w, h = [int(v) for v in bbox]
initial_bbox_size = (w, h) # Store initial size

# 5. Ініціалізуємо CSRT
tracker.init(first_frame, bbox)
P_prev = np.array([x, y], dtype=np.float32) # Previous ABSOLUTE top-left position of the object

# --- Kalman Filter Initialisation (for RELATIVE position) ---
kf = cv2.KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE)
# Transition Matrix: Tracks state based on current RELATIVE velocity
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(STATE_SIZE, dtype=np.float32) * KF_PROCESS_NOISE_COV
kf.measurementNoiseCov = np.eye(MEASUREMENT_SIZE, dtype=np.float32) * KF_MEASUREMENT_NOISE_COV

# Initial RELATIVE state: (0, 0) relative shift, (0, 0) relative velocity
kf.statePost = np.array([[0.], [0.], [0.], [0.]], np.float32)
measurement = np.zeros((MEASUREMENT_SIZE, 1), np.float32) 

# --- Global Flow Initialisation ---
# Find initial key points for background motion estimation (excluding the bird area)
mask_global = np.ones_like(old_gray, dtype=np.uint8) * 255
# Mask out the bird to ensure we only track background features
mask_global[y:y+h, x:x+w] = 0 
global_points_prev = cv2.goodFeaturesToTrack(old_gray, mask=mask_global, 
                                             maxCorners=FLOW_MAX_CORNERS, 
                                             qualityLevel=FLOW_QUALITY, 
                                             minDistance=FLOW_MIN_DISTANCE)

# 6. Готуємося до циклу
gif_frames_csrt = []
WINDOW_TRACKER = "Hybrid CSRT + Kalman (Motion Compensated)"

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of tracking.")
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    # 7. GLOBAL MOTION COMPENSATION (Background movement estimation)
    global_dx, global_dy = 0.0, 0.0
    
    if global_points_prev is not None and len(global_points_prev) > 10:
        # Track background features
        global_points_curr, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, global_points_prev, None, **FLOW_LK_PARAMS)
        
        good_curr = global_points_curr[status == 1]
        good_prev = global_points_prev[status == 1]
        
        if len(good_curr) > 10:
            # Estimate global shift using median displacement
            global_dx = np.median(good_curr[:, 0] - good_prev[:, 0])
            global_dy = np.median(good_curr[:, 1] - good_prev[:, 1])
            
            # Update background features for the next frame (optional: re-detect periodically)
            global_points_prev = good_curr.reshape(-1, 1, 2)
            
    # Calculate the ABSOLUTE expected position based on background motion
    P_expected = P_prev + np.array([global_dx, global_dy], dtype=np.float32)
    
    # 8. KALMAN FILTER PREDICTION
    kf_prediction = kf.predict()
    
    # 9. CSRT Update
    ok, new_bbox_csrt = tracker.update(frame)
    
    # 10. CORRECTION and DECISION
    success_flag = ok
    P_new = P_prev # Default to old position in case of full failure

    if success_flag:
        # CSRT succeeded, calculate the RELATIVE shift from the expected position
        new_x, new_y = float(new_bbox_csrt[0]), float(new_bbox_csrt[1])
        P_new = np.array([new_x, new_y], dtype=np.float32)
        
        relative_shift = P_new - P_expected
        
        # Correct the Kalman Filter using the RELATIVE shift
        measurement[0] = relative_shift[0]
        measurement[1] = relative_shift[1]
        kf_corrected = kf.correct(measurement)
        
        # Get the corrected RELATIVE state
        corrected_relative_x = kf_corrected[0][0]
        corrected_relative_y = kf_corrected[1][0]
        
        # Calculate the final ABSOLUTE position for drawing
        Final_P_abs = P_expected + np.array([corrected_relative_x, corrected_relative_y], dtype=np.float32)
        
        cv2.putText(frame, "Tracking", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
    else:
        # CSRT failed. Rely purely on Kalman's prediction of RELATIVE shift.
        
        predicted_relative_x = kf_prediction[0][0]
        predicted_relative_y = kf_prediction[1][0]
        
        # Calculate the final ABSOLUTE position based on expected background motion + predicted relative motion
        Final_P_abs = P_expected + np.array([predicted_relative_x, predicted_relative_y], dtype=np.float32)
        
        # RE-INITIALIZE CSRT at the predicted location
        final_x, final_y = Final_P_abs[0], Final_P_abs[1]
        bbox_reinit = (final_x, final_y, w, h)
        tracker.init(frame, bbox_reinit)
        
        cv2.putText(frame, "Kalman Prediction Re-init", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)


    # 11. Малюємо рамку та оновлюємо стан
    
    # Update P_prev for the next iteration
    P_prev = Final_P_abs 
    
    # Prepare BBOX for drawing
    bbox_to_draw = (Final_P_abs[0], Final_P_abs[1], initial_bbox_size[0], initial_bbox_size[1])

    p1 = (int(bbox_to_draw[0]), int(bbox_to_draw[1]))
    p2 = (int(bbox_to_draw[0] + bbox_to_draw[2]), int(bbox_to_draw[1] + bbox_to_draw[3]))
    color = (0, 255, 0) if success_flag else (0, 0, 255) 
    
    cv2.rectangle(frame, p1, p2, color, 6, 1)
    cv2.putText(frame, "Tracking Success" if success_flag else "Tracking Failure", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Додаємо кадр для GIF
    gif_frames_csrt.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow(WINDOW_TRACKER, frame)
    
    if len(gif_frames_csrt) == 1:
        cv2.namedWindow(WINDOW_TRACKER)
        set_window_position(WINDOW_TRACKER, WINDOW_START_X, WINDOW_START_Y)
    
    # Update grayscale frame for next global motion estimation
    old_gray = frame_gray.copy() 
    
    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
        
# 12. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_motion_comp_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    
    print(f"Creating motion compensated tracking GIF with {len(gif_frames_csrt)} frames...")
    create_gif(gif_frames_csrt, output_path_csrt, fps=10) 

# 13. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("Motion Compensated Tracking finished.")
'''
'''
#--------------------------------------------------------------------------------#
#----------Part Y: Hybrid CSRT + Kalman (Motion Compensation) + YOLOv5-----------#
#--------------------------------------------------------------------------------#

# --- Global Window Position ---
WINDOW_START_X = 50
WINDOW_START_Y = 50

# --- CONFIGURATION PARAMETERS ---
# YOLO parameters
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence to consider a detection as a bird
REDETECTION_INTERVAL = 7     # Run YOLO detector every 15 frames
IOU_THRESHOLD = 0.4           # IoU threshold to decide if tracker has drifted

# CSRT parameters
USE_HOG_FEATURES = True
HISTOGRAM_BINS = 32

# Kalman Filter parameters
STATE_SIZE = 4
MEASUREMENT_SIZE = 2
KF_PROCESS_NOISE_COV = 0.05
KF_MEASUREMENT_NOISE_COV = 2

# Global Flow parameters
FLOW_MAX_CORNERS = 500
FLOW_QUALITY = 0.01
FLOW_MIN_DISTANCE = 30
FLOW_LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# --- Helper Function for Window Position ---
def set_window_position(window_name, x, y):
    cv2.moveWindow(window_name, x, y)

# --- Helper function for IoU calculation ---
def calculate_iou(boxA, boxB):
    # box format: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# -------------------------- MAIN SCRIPT ---------------------------

print("\nStarting Part Y: Hybrid CSRT + Kalman + YOLOv5 Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_csrt = VIDEO
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# --- YOLOv5 Model Loading ---
print("Loading YOLOv5 model... This might take a moment on the first run.")
# Use a pre-trained model. 'yolov5s' is small and fast.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = CONFIDENCE_THRESHOLD
# COCO class for 'bird' is 14
BIRD_CLASS_INDEX = 14
print("YOLOv5 model loaded successfully.")


# 2. Створюємо об'єкт трекера CSRT
params = cv2.TrackerCSRT_Params()
params.use_hog = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)

# 3. Зчитуємо перший кадр
ret, first_frame = video_csrt.read()
if not ret:
    print("Failed to read video")
    exit()

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
FRAME_H, FRAME_W, _ = first_frame.shape

# 4. Дозволяємо користувачу вибрати об'єкт (з допомогою YOLO)
WINDOW_ROI = "Select Bird to Track"
cv2.namedWindow(WINDOW_ROI)
set_window_position(WINDOW_ROI, WINDOW_START_X, WINDOW_START_Y)

print("Running initial detection with YOLO...")
results = model(first_frame)
detections = results.xyxy[0].cpu().numpy()
bird_detections = [d for d in detections if int(d[5]) == BIRD_CLASS_INDEX]

if bird_detections:
    print(f"Found {len(bird_detections)} bird(s). Please select one.")
    # Fallback to manual selection if YOLO fails or user wants to override
    bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)
else:
    print("YOLO did not find a bird. Please select the object manually.")
    bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)

cv2.destroyWindow(WINDOW_ROI)

x, y, w, h = [int(v) for v in bbox]
initial_bbox_size = (w, h)

# 5. Ініціалізуємо CSRT
tracker.init(first_frame, bbox)
P_prev = np.array([x, y], dtype=np.float32)

# --- Kalman Filter Initialisation ---
kf = cv2.KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(STATE_SIZE, dtype=np.float32) * KF_PROCESS_NOISE_COV
kf.measurementNoiseCov = np.eye(MEASUREMENT_SIZE, dtype=np.float32) * KF_MEASUREMENT_NOISE_COV
kf.statePost = np.array([[0.], [0.], [0.], [0.]], np.float32)
measurement = np.zeros((MEASUREMENT_SIZE, 1), np.float32)

# --- Global Flow Initialisation ---
mask_global = np.ones_like(old_gray, dtype=np.uint8) * 255
mask_global[y:y+h, x:x+w] = 0
global_points_prev = cv2.goodFeaturesToTrack(old_gray, mask=mask_global, maxCorners=FLOW_MAX_CORNERS, qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DISTANCE)

# 6. Готуємося до циклу
gif_frames_csrt = []
frame_count = 0
WINDOW_TRACKER = "Hybrid CSRT + Kalman + YOLO (Corrected)"

while True:
    ret, frame = video_csrt.read()
    if not ret:
        print("End of tracking.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # --- PERIODIC RE-DETECTION WITH YOLO ---
    if frame_count % REDETECTION_INTERVAL == 0:
        yolo_results = model(frame)
        yolo_detections = yolo_results.xyxy[0].cpu().numpy()
        yolo_bird_detections = [d for d in yolo_detections if int(d[5]) == BIRD_CLASS_INDEX]

        if yolo_bird_detections:
            # Find the detection closest to the tracker's last known position
            tracker_bbox = (P_prev[0], P_prev[1], initial_bbox_size[0], initial_bbox_size[1])
            best_iou = 0
            best_detection = None

            for det in yolo_bird_detections:
                yolo_box = (int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1]))
                iou = calculate_iou(tracker_bbox, yolo_box)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = yolo_box

            # If a good match is found, check if re-initialization is needed
            if best_detection and best_iou > IOU_THRESHOLD:
                print(f"Frame {frame_count}: YOLO correction. IoU: {best_iou:.2f}")
                # Re-initialize tracker and Kalman at YOLO's position
                tracker = cv2.TrackerCSRT_create(params)
                tracker.init(frame, best_detection)
                P_prev = np.array([best_detection[0], best_detection[1]], dtype=np.float32)
                # Reset Kalman filter state
                kf.statePost = np.array([[0.], [0.], [0.], [0.]], np.float32)
                cv2.putText(frame, "YOLO Re-Init", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


    # 7. GLOBAL MOTION COMPENSATION
    global_dx, global_dy = 0.0, 0.0
    if global_points_prev is not None and len(global_points_prev) > 10:
        global_points_curr, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, global_points_prev, None, **FLOW_LK_PARAMS)
        good_curr = global_points_curr[status == 1]
        good_prev = global_points_prev[status == 1]
        if len(good_curr) > 10:
            global_dx = np.median(good_curr[:, 0] - good_prev[:, 0])
            global_dy = np.median(good_curr[:, 1] - good_prev[:, 1])
            global_points_prev = good_curr.reshape(-1, 1, 2)
    P_expected = P_prev + np.array([global_dx, global_dy], dtype=np.float32)

    # 8. KALMAN FILTER PREDICTION
    kf_prediction = kf.predict()

    # 9. CSRT Update
    ok, new_bbox_csrt = tracker.update(frame)

    # 10. CORRECTION and DECISION
    success_flag = ok
    P_new = P_prev
    if success_flag:
        new_x, new_y = float(new_bbox_csrt[0]), float(new_bbox_csrt[1])
        P_new = np.array([new_x, new_y], dtype=np.float32)
        relative_shift = P_new - P_expected
        measurement[0] = relative_shift[0]
        measurement[1] = relative_shift[1]
        kf_corrected = kf.correct(measurement)
        corrected_relative_x = kf_corrected[0][0]
        corrected_relative_y = kf_corrected[1][0]
        Final_P_abs = P_expected + np.array([corrected_relative_x, corrected_relative_y], dtype=np.float32)
        cv2.putText(frame, "Tracking", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        predicted_relative_x = kf_prediction[0][0]
        predicted_relative_y = kf_prediction[1][0]
        Final_P_abs = P_expected + np.array([predicted_relative_x, predicted_relative_y], dtype=np.float32)
        final_x, final_y = Final_P_abs[0], Final_P_abs[1]
        w, h = initial_bbox_size
        bbox_reinit = (final_x, final_y, w, h)
        # We don't re-init with Kalman alone anymore, we wait for YOLO
        # tracker.init(frame, bbox_reinit)
        cv2.putText(frame, "CSRT Lost! Waiting for YOLO", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 11. Малюємо рамку та оновлюємо стан
    P_prev = Final_P_abs
    bbox_to_draw = (Final_P_abs[0], Final_P_abs[1], initial_bbox_size[0], initial_bbox_size[1])
    p1 = (int(bbox_to_draw[0]), int(bbox_to_draw[1]))
    p2 = (int(bbox_to_draw[0] + bbox_to_draw[2]), int(bbox_to_draw[1] + bbox_to_draw[3]))
    color = (0, 255, 0) if success_flag else (0, 0, 255)
    cv2.rectangle(frame, p1, p2, color, 6, 1)
    cv2.putText(frame, "Tracking Success" if success_flag else "Tracking Failure", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    gif_frames_csrt.append(frame)
    cv2.imshow(WINDOW_TRACKER, frame)
    if len(gif_frames_csrt) == 1:
        cv2.namedWindow(WINDOW_TRACKER)
        set_window_position(WINDOW_TRACKER, WINDOW_START_X, WINDOW_START_Y)

    old_gray = frame_gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 12. Зберігаємо результат у GIF
if gif_frames_csrt:
    gif_filename_csrt = 'hybrid_yolo_corrected_tracking.gif'
    output_path_csrt = os.path.join(output_folder, gif_filename_csrt)
    print(f"Creating YOLO corrected tracking GIF with {len(gif_frames_csrt)} frames...")
    create_gif(gif_frames_csrt, output_path_csrt, fps=10)

# 13. Звільняємо ресурси
video_csrt.release()
cv2.destroyAllWindows()
print("YOLO Corrected Tracking finished.")
'''

#------------------------------------------------------------------------#
# Hybrid tracker: CSRT + Kalman (long-horizon prediction)
#   + similarity score + confidence gating
#   + GLOBAL MOTION COMPENSATION (affine, ORB+RANSAC)
#   + MOTION GATING (Farnebäck optical flow)
#   + ANTI-DISTRACTOR (LAB-L, edge density, foreground ratio)
#------------------------------------------------------------------------#

# --- CSRT PARAMETERS (під важкий фон/краї) ---
USE_HOG_FEATURES = False
HISTOGRAM_BINS   = 32

# --- Kalman + Fusion PARAMETERS ---
DT = 1.0
PREDICT_AHEAD_FRAMES  = 6
SIM_THRESHOLD_WEAK    = 0.40
SIM_THRESHOLD_GOOD    = 0.60
CONSISTENCY_MAX_PX    = 100
CONF_LOSS_THRESHOLD   = 0.50
CONF_STRONG_THRESHOLD = 0.70
CONF_BLEND_SIM        = 0.55   # менше довіри до кольору/градієнтів, більше до руху/кінематики

# --- Template Matching PARAMETERS ---
TM_METHOD       = cv2.TM_CCOEFF_NORMED
TM_SEARCH_SCALE = 2.5
TM_MIN_SCORE    = 0.50

# --- Global Motion (ego-motion) PARAMETERS ---
GMC_USE                = True
ORB_N_FEATURES         = 800
ORB_FAST_THRESHOLD     = 7
GMC_MIN_INLIERS        = 15
GMC_RANSAC_REPROJ_THR  = 3.0

# --- Motion gating (Farnebäck) ---
FLOW_PYR_SCALE  = 0.5
FLOW_LEVELS     = 3
FLOW_WINSIZE    = 15
FLOW_ITERS      = 3
FLOW_POLY_N     = 5
FLOW_POLY_SIGMA = 1.2

MOTION_NORM_PX        = 8.0    # середня швидкість пікселя, що вважається «насиченою»
MOTION_MIN_FOR_ACCEPT = 0.20   # мін. рух у bbox для прийняття

# --- Anti-distractor (light/texture/fg) ---
LAB_L_MAX_Z     = 2.0          # скільки σ дозволяємо відхилитись по L (LAB)
EDGE_DENS_MIN   = 0.05         # мін. щільність ребер у bbox
FG_RATIO_MIN    = 0.15         # мін. частка foreground у bbox
W_MOT           = 0.35         # вага руху у CONF
W_FG            = 0.25         # вага foreground у CONF
W_EDGE          = 0.15         # вага текстури у CONF
W_L             = 0.20         # штраф за відхилення яскравості
AD_HARD_REJECT  = True         # увімкнути жорсткий гейтинг проти «світлих плям»

# --- Drawing ---
BOX_THICK = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Global Window Position & Sizes ---
WINDOW_START_X = 50
WINDOW_START_Y = 50
ROI_WINDOW_W   = 1200
ROI_WINDOW_H   = 800
MAIN_WINDOW_W  = 1280
MAIN_WINDOW_H  = 720

# ------------------------ Utility functions ----------------------------

def bbox_to_cxcywh(b):
    x, y, w, h = b
    return np.array([x + w/2.0, y + h/2.0, w, h], dtype=np.float32)

def cxcywh_to_bbox(c):
    cx, cy, w, h = c
    return (int(cx - w/2.0), int(cy - h/2.0), int(w), int(h))

def clip_bbox_to_frame(bbox, frame_shape):
    H, W = frame_shape[:2]
    x, y, w, h = bbox
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)

def extract_patch(image, bbox):
    x, y, w, h = bbox
    H, W = image.shape[:2]
    x2, y2 = x + w, y + h
    if x < 0 or y < 0 or x2 > W or y2 > H:
        return None
    return image[y:y+h, x:x+w].copy()

def hsv_hist_corr(patch, bins=64):
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def compare_hists(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

def motion_consistency(pred_cx, pred_cy, meas_cx, meas_cy, norm_px=80.0):
    d = np.hypot(pred_cx - meas_cx, pred_cy - meas_cy)
    return float(max(0.0, 1.0 - d / max(1.0, norm_px)))

def compute_confidence(sim, cons, w_sim=0.7):
    sim_clip = np.clip(sim, 0.0, 1.0)
    cons_clip = np.clip(cons, 0.0, 1.0)
    return float(w_sim * sim_clip + (1.0 - w_sim) * cons_clip)

def template_match_reacquire(frame, template, pred_bbox, scale=2.0, method=cv2.TM_CCOEFF_NORMED):
    H, W = frame.shape[:2]
    px, py, pw, ph = pred_bbox
    cx, cy = int(px + pw/2), int(py + ph/2)
    win_w = int(pw * scale)
    win_h = int(ph * scale)
    x0 = max(0, cx - win_w//2)
    y0 = max(0, cy - win_h//2)
    x1 = min(W, x0 + win_w)
    y1 = min(H, y0 + win_h)
    search = frame[y0:y1, x0:x1]
    if search.size == 0 or template is None or template.size == 0:
        return None, 0.0
    th, tw = template.shape[:2]
    if th >= search.shape[0] or tw >= search.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(search, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    rx, ry = max_loc
    bb = (x0 + rx, y0 + ry, tw, th)
    return bb, float(max_val)

# ------------------------ Global motion (affine) ------------------------

def estimate_global_affine(prev_gray, gray):
    try:
        orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES, fastThreshold=ORB_FAST_THRESHOLD)
        k1, d1 = orb.detectAndCompute(prev_gray, None)
        k2, d2 = orb.detectAndCompute(gray, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(d1, d2)
        if len(matches) < 8:
            return np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        matches = sorted(matches, key=lambda m: m.distance)
        N = min(200, len(matches))
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches[:N]])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches[:N]])
        A, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2, method=cv2.RANSAC,
            ransacReprojThreshold=GMC_RANSAC_REPROJ_THR,
            maxIters=2000, confidence=0.99
        )
        if A is None:
            return np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        if inliers is not None and np.count_nonzero(inliers) >= GMC_MIN_INLIERS:
            return A.astype(np.float32)
        return np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    except Exception:
        return np.array([[1,0,0],[0,1,0]], dtype=np.float32)

def warp_point_affine(x, y, A):
    vec = np.array([x, y, 1.0], dtype=np.float32)
    res = A @ vec
    return float(res[0]), float(res[1])

def warp_bbox_affine(bbox, A):
    x, y, w, h = bbox
    cx = x + w/2.0
    cy = y + h/2.0
    cx2, cy2 = warp_point_affine(cx, cy, A)
    a11, a12, _ = A[0]; a21, a22, _ = A[1]
    sx = np.sqrt(a11*a11 + a21*a21); sy = np.sqrt(a12*a12 + a22*a22)
    if sx <= 1e-6: sx = 1.0
    if sy <= 1e-6: sy = 1.0
    w2 = int(max(1, round(w * sx)))
    h2 = int(max(1, round(h * sy)))
    x2 = int(round(cx2 - w2/2.0))
    y2 = int(round(cy2 - h2/2.0))
    return (x2, y2, w2, h2)

# ------------------------ Motion gating utils --------------------------

def compute_flow_mag(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=FLOW_PYR_SCALE, levels=FLOW_LEVELS,
        winsize=FLOW_WINSIZE, iterations=FLOW_ITERS,
        poly_n=FLOW_POLY_N, poly_sigma=FLOW_POLY_SIGMA, flags=0
    )
    mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    return mag

def motion_score_from_mag(mag, bbox, norm_px=MOTION_NORM_PX):
    H, W = mag.shape
    x, y, w, h = clip_bbox_to_frame(bbox, (H, W, 1))
    patch = mag[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0
    return float(np.clip(np.mean(patch) / max(1e-6, norm_px), 0.0, 1.0))

# ------------------------ Anti-distractor utils ------------------------

def bbox_lab_stats(image_bgr, bbox):
    x,y,w,h = clip_bbox_to_frame(bbox, image_bgr.shape)
    patch = image_bgr[y:y+h, x:x+w]
    if patch is None or patch.size == 0:
        return None, None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    L = lab[...,0].astype(np.float32)  # [0..255]
    return float(np.mean(L)), float(np.std(L) + 1e-6)

def edge_density(image_gray, bbox):
    x,y,w,h = clip_bbox_to_frame(bbox, (image_gray.shape[0], image_gray.shape[1], 1))
    roi = image_gray[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    v = np.median(roi)
    lo = int(max(0, 0.66*v))
    hi = int(min(255, 1.33*v + 30))
    edges = cv2.Canny(roi, lo, hi, L2gradient=True)
    return float(np.count_nonzero(edges)) / float(max(1, w*h))

def fg_ratio_from_mask(fgmask, bbox):
    x,y,w,h = clip_bbox_to_frame(bbox, (fgmask.shape[0], fgmask.shape[1], 1))
    roi = fgmask[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    fg = np.count_nonzero(roi == 255)  # тіні (127) ігноруємо
    return float(fg) / float(max(1, w*h))

# ------------------------ Kalman construction --------------------------

def build_kalman(dt=1.0):
    # State: [cx, cy, vx, vy, w, h]^T ; Meas: [cx, cy, w, h]^T
    kf = cv2.KalmanFilter(6, 4, 0, cv2.CV_32F)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1,  0, 0, 0],
        [0, 0, 0,  1, 0, 0],
        [0, 0, 0,  0, 1, 0],
        [0, 0, 0,  0, 0, 1]
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=np.float32)
    kf.processNoiseCov     = np.diag([1e-2, 1e-2, 1e-1, 1e-1, 1e-3, 1e-3]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.5e-2, 2.5e-2, 1e-2, 1e-2]).astype(np.float32)
    kf.errorCovPost        = np.diag([1, 1, 10, 10, 1, 1]).astype(np.float32)
    return kf

# ----------------------------- Main logic ------------------------------

print("\nStarting Hybrid Tracking (CSRT + Kalman + Similarity + Confidence + GMC + MOTION + AntiDistractor)...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# fps -> DT
_fps = VIDEO.get(cv2.CAP_PROP_FPS) if hasattr(VIDEO, "get") else 0.0
if _fps and _fps > 1e-3:
    DT = 1.0 / _fps

video = VIDEO
ret, first_frame = video.read()
if not ret:
    print("Failed to read video")
    raise SystemExit

# ROI window (кероване вікно)
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.moveWindow("Select ROI", WINDOW_START_X, WINDOW_START_Y)
cv2.resizeWindow("Select ROI", ROI_WINDOW_W, ROI_WINDOW_H)
cv2.imshow("Select ROI", first_frame)

print("Please select a bounding box for the object and press ENTER or SPACE.")
init_bbox = cv2.selectROI("Select ROI", first_frame, False)
cv2.destroyWindow("Select ROI")
if init_bbox is None or init_bbox[2] <= 1 or init_bbox[3] <= 1:
    print("Invalid ROI")
    raise SystemExit

# CSRT init
params = cv2.TrackerCSRT_Params()
params.use_hog       = USE_HOG_FEATURES
params.histogram_bins= HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)
tracker.init(first_frame, init_bbox)

# Kalman init
kf = build_kalman(DT)
cx, cy, w, h = bbox_to_cxcywh(init_bbox)
kf.statePost = np.array([[cx],[cy],[0.0],[0.0],[w],[h]], dtype=np.float32)

# Template & hist
template0 = extract_patch(first_frame, clip_bbox_to_frame(init_bbox, first_frame.shape))
hist0 = hsv_hist_corr(template0, bins=HISTOGRAM_BINS)

# Anti-distractor reference (LAB-L) + BG subtractor
L0_mean, L0_std = bbox_lab_stats(first_frame, init_bbox)
bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
_ = bg_sub.apply(first_frame)

# Main window
cv2.namedWindow("Hybrid CSRT+Kalman", cv2.WINDOW_NORMAL)
cv2.moveWindow("Hybrid CSRT+Kalman", WINDOW_START_X, WINDOW_START_Y + 60)
cv2.resizeWindow("Hybrid CSRT+Kalman", MAIN_WINDOW_W, MAIN_WINDOW_H)

gif_frames   = []
conf_history = deque(maxlen=50)
pred_ahead_left = 0
frame_idx = 0

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video.")
        break

    frame_vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 0) GMC (prev_gray -> gray)
    A = estimate_global_affine(prev_gray, gray) if GMC_USE else np.array([[1,0,0],[0,1,0]], dtype=np.float32)

    # 0a) Flow mag + FG mask (для motion/fg метрик)
    mag    = compute_flow_mag(prev_gray, gray)
    fgmask = bg_sub.apply(frame)

    # 1) Kalman predict (+ перенос у координати поточного кадру)
    kf.predict()
    pred_cx, pred_cy, _, _, pred_w, pred_h = kf.statePre.flatten()
    pred_bbox     = clip_bbox_to_frame(cxcywh_to_bbox((pred_cx, pred_cy, pred_w, pred_h)), frame.shape)
    pred_bbox_aff = clip_bbox_to_frame(warp_bbox_affine(pred_bbox, A), frame.shape)

    had_measurement = False
    used_reacquire  = False
    final_bbox = None

    # 2) CSRT update
    csrt_ok, csrt_bbox = tracker.update(frame)

    if csrt_ok:
        csrt_bbox = clip_bbox_to_frame(tuple(map(int, csrt_bbox)), frame.shape)
        csrt_patch = extract_patch(frame, csrt_bbox)
        hist_t = hsv_hist_corr(csrt_patch, bins=HISTOGRAM_BINS)
        sim = compare_hists(hist0, hist_t)

        # consistency
        m_cx, m_cy, m_w, m_h = bbox_to_cxcywh(csrt_bbox)
        p_cx, p_cy, _, _     = bbox_to_cxcywh(pred_bbox_aff)
        cons = motion_consistency(p_cx, p_cy, m_cx, m_cy, norm_px=CONSISTENCY_MAX_PX)

        # motion / anti-distractor
        mot     = motion_score_from_mag(mag, csrt_bbox, norm_px=MOTION_NORM_PX)
        L_m, L_s= bbox_lab_stats(frame, csrt_bbox)
        edens   = edge_density(gray, csrt_bbox)
        fgr     = fg_ratio_from_mask(fgmask, csrt_bbox)
        zL      = abs((L_m - L0_mean) / max(1e-6, L0_std)) if L_m is not None else 999.0

        conf_no_mot = compute_confidence(sim, cons, w_sim=CONF_BLEND_SIM)
        conf = float((1.0 - W_MOT)*conf_no_mot + W_MOT*mot + W_FG*fgr + W_EDGE*edens - W_L*min(1.0, zL/LAB_L_MAX_Z))
        conf_history.append(conf)

        hard_reject = AD_HARD_REJECT and ((fgr < FG_RATIO_MIN) or (edens < EDGE_DENS_MIN) or (zL > LAB_L_MAX_Z))

        if hard_reject or conf < CONF_LOSS_THRESHOLD or sim < SIM_THRESHOLD_WEAK or mot < MOTION_MIN_FOR_ACCEPT:
            # локальний пошук біля прогнозу
            reacq_bbox, tm_score = template_match_reacquire(frame, template0, pred_bbox_aff,
                                                            scale=TM_SEARCH_SCALE, method=TM_METHOD)
            if reacq_bbox is not None:
                mot_r   = motion_score_from_mag(mag, reacq_bbox, norm_px=MOTION_NORM_PX)
                Lr_m, Lr_s = bbox_lab_stats(frame, reacq_bbox)
                edens_r = edge_density(gray, reacq_bbox)
                fgr_r   = fg_ratio_from_mask(fgmask, reacq_bbox)
                zL_r    = abs((Lr_m - L0_mean) / max(1e-6, L0_std)) if Lr_m is not None else 999.0
            else:
                mot_r = edens_r = fgr_r = tm_score = 0.0
                zL_r = 999.0

            accept_reacq = (reacq_bbox is not None and tm_score >= TM_MIN_SCORE and
                            mot_r >= MOTION_MIN_FOR_ACCEPT and fgr_r >= FG_RATIO_MIN and
                            edens_r >= EDGE_DENS_MIN and zL_r <= LAB_L_MAX_Z)

            if accept_reacq:
                rx, ry, rw, rh = clip_bbox_to_frame(reacq_bbox, frame.shape)
                meas = np.array([[rx + rw/2.0],[ry + rh/2.0],[rw],[rh]], dtype=np.float32)
                kf.correct(meas)
                final_bbox = (rx, ry, rw, rh)
                used_reacquire = True
                had_measurement = True
                pred_ahead_left = 0
                tracker = cv2.TrackerCSRT_create(params)
                tracker.init(frame, final_bbox)
                cv2.putText(frame_vis, f"Reacq TM:{tm_score:.2f} MOT:{mot_r:.2f} Lz:{zL_r:.1f} ED:{edens_r:.2f} FG:{fgr_r:.2f}",
                            (20, 70), FONT, 0.6, (0,255,255), 2, cv2.LINE_AA)
            else:
                if pred_ahead_left == 0:
                    pred_ahead_left = PREDICT_AHEAD_FRAMES
                pred_ahead_left = max(0, pred_ahead_left - 1)
                final_bbox = pred_bbox_aff
                had_measurement = False
                cv2.putText(frame_vis, "Predict-only (Kalman+GMC)", (20, 70), FONT, 0.7, (0,165,255), 2, cv2.LINE_AA)
        else:
            meas = np.array([[m_cx],[m_cy],[m_w],[m_h]], dtype=np.float32)
            kf.correct(meas)
            final_bbox = csrt_bbox
            had_measurement = True
            pred_ahead_left = 0

        color = (0,255,0) if had_measurement and not used_reacquire else ((255,0,0) if used_reacquire else (0,255,255))
        cv2.rectangle(frame_vis,
                      (final_bbox[0], final_bbox[1]),
                      (final_bbox[0]+final_bbox[2], final_bbox[1]+final_bbox[3]),
                      color, BOX_THICK, 1)
        cv2.putText(frame_vis,
                    f"SIM:{sim:.2f} CONS:{cons:.2f} MOT:{mot:.2f} Lz:{zL:.1f} ED:{edens:.2f} FG:{fgr:.2f} CONF:{conf:.2f}",
                    (20, 40), FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)

        # Обережне оновлення шаблону (щоб не «доучитись» на фон)
        if conf >= CONF_STRONG_THRESHOLD and mot >= MOTION_MIN_FOR_ACCEPT and \
           edens >= EDGE_DENS_MIN and fgr >= FG_RATIO_MIN and zL <= LAB_L_MAX_Z:
            template0 = extract_patch(frame, final_bbox)
            hist0 = hsv_hist_corr(template0, bins=HISTOGRAM_BINS)
            # повільно «підтягуємо» еталонну яскравість
            if L_m is not None:
                L0_mean = 0.9*L0_mean + 0.1*L_m
                L0_std  = max(1.0, 0.9*L0_std + 0.1*(L_s if L_s is not None else L0_std))

    else:
        # CSRT failed -> спроба пере-захоплення з фільтрами проти дистракторів
        reacq_bbox, tm_score = template_match_reacquire(frame, template0, pred_bbox_aff,
                                                        scale=TM_SEARCH_SCALE, method=TM_METHOD)
        if reacq_bbox is not None:
            mot_r   = motion_score_from_mag(mag, reacq_bbox, norm_px=MOTION_NORM_PX)
            Lr_m, Lr_s = bbox_lab_stats(frame, reacq_bbox)
            edens_r = edge_density(gray, reacq_bbox)
            fgr_r   = fg_ratio_from_mask(fgmask, reacq_bbox)
            zL_r    = abs((Lr_m - L0_mean) / max(1e-6, L0_std)) if Lr_m is not None else 999.0
        else:
            mot_r = edens_r = fgr_r = tm_score = 0.0
            zL_r = 999.0

        accept_reacq = (reacq_bbox is not None and tm_score >= TM_MIN_SCORE and
                        mot_r >= MOTION_MIN_FOR_ACCEPT and fgr_r >= FG_RATIO_MIN and
                        edens_r >= EDGE_DENS_MIN and zL_r <= LAB_L_MAX_Z)

        if accept_reacq:
            rx, ry, rw, rh = clip_bbox_to_frame(reacq_bbox, frame.shape)
            meas = np.array([[rx + rw/2.0],[ry + rh/2.0],[rw],[rh]], dtype=np.float32)
            kf.correct(meas)
            final_bbox = (rx, ry, rw, rh)
            used_reacquire = True
            had_measurement = True
            pred_ahead_left = 0
            tracker = cv2.TrackerCSRT_create(params)
            tracker.init(frame, final_bbox)
            cv2.putText(frame_vis, f"Reacquired TM:{tm_score:.2f} MOT:{mot_r:.2f} Lz:{zL_r:.1f} ED:{edens_r:.2f} FG:{fgr_r:.2f}",
                        (20, 40), FONT, 0.6, (0,255,255), 2, cv2.LINE_AA)
            # можна одразу освіжити шаблон, якщо сигнал якісний
            if mot_r >= MOTION_MIN_FOR_ACCEPT and edens_r >= EDGE_DENS_MIN and fgr_r >= FG_RATIO_MIN and zL_r <= LAB_L_MAX_Z and tm_score >= (TM_MIN_SCORE + 0.05):
                template0 = extract_patch(frame, final_bbox)
                hist0 = hsv_hist_corr(template0, bins=HISTOGRAM_BINS)
        else:
            if pred_ahead_left == 0:
                pred_ahead_left = PREDICT_AHEAD_FRAMES
            pred_ahead_left = max(0, pred_ahead_left - 1)
            final_bbox = pred_bbox_aff
            had_measurement = False
            cv2.putText(frame_vis, "Predict-only (Kalman+GMC)", (20, 40), FONT, 0.7, (0,165,255), 2, cv2.LINE_AA)

        cv2.rectangle(frame_vis,
                      (final_bbox[0], final_bbox[1]),
                      (final_bbox[0]+final_bbox[2], final_bbox[1]+final_bbox[3]),
                      (0,165,255), BOX_THICK, 1)

    # Overlay прогнозу Калмана (після GMC)
    cv2.rectangle(frame_vis,
                  (pred_bbox_aff[0], pred_bbox_aff[1]),
                  (pred_bbox_aff[0]+pred_bbox_aff[2], pred_bbox_aff[1]+pred_bbox_aff[3]),
                  (200,200,200), 2, 1)
    cv2.putText(frame_vis, "Kalman pred (GMC)",
                (pred_bbox_aff[0], max(0, pred_bbox_aff[1]-8)),
                FONT, 0.6, (200,200,200), 1, cv2.LINE_AA)

    # Show + (optional) GIF buffer
    cv2.imshow("Hybrid CSRT+Kalman", frame_vis)
    gif_frames.append(frame_vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    frame_idx += 1
    prev_gray = gray

# --- Save GIF (optional) ---
try:
    if len(gif_frames) > 0 and 'create_gif' in globals():
        out_gif = os.path.join(OUTPUT_FOLDER, "hybrid_csrt_kalman_gmc_motion_antidistr.gif")
        print(f"Creating GIF with {len(gif_frames)} frames...")
        create_gif(gif_frames, out_gif, fps=int(_fps) if _fps and _fps > 1 else 10)
except Exception as e:
    print(f"(Skipping GIF) Reason: {e}")

video.release()
cv2.destroyAllWindows()
print("Hybrid tracking finished.")
