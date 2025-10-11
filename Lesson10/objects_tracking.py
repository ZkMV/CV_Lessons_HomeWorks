import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import imageio
import os
#from screeninfo import get_monitors

# Центральна константа для збереження результатів
OUTPUT_FOLDER = os.path.join('Lesson10', 'result')

#video1 = cv2.VideoCapture('Lesson10/source/birds_in_sky.mp4')
#video1 = cv2.VideoCapture('source//hawk.mp6')
output_folder = OUTPUT_FOLDER

#VIDEO = cv2.VideoCapture('Lesson10/source/birds_in_sky.mp4')
VIDEO = cv2.VideoCapture('Lesson10/source/hawk.mp4')

# --- Global Window Position ---
# Set a fixed, visible position for the windows
WINDOW_START_X = 50
WINDOW_START_Y = 50
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


'''
#------------------------------------------------------------------------#
#----------------------------- 1: KCF--------------------------------#
#------------------------------------------------------------------------#
print("\nStarting 1: KCF Tracking...")
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

FRAME_H, FRAME_W, _ = first_frame.shape 

# 4. Дозволяємо користувачу вибрати об'єкт для відстеження
WINDOW_ROI = "Select Bird to Track"
cv2.namedWindow(WINDOW_ROI)
# Set fixed position for the ROI selection window
cv2.moveWindow(WINDOW_ROI, WINDOW_START_X, WINDOW_START_Y) 

print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)
cv2.destroyWindow(WINDOW_ROI)

# 5. Ініціалізуємо трекер першим кадром та вибраною рамкою
tracker.init(first_frame, bbox)

# 6. Готуємося до циклу
gif_frames_kcf = []
WINDOW_TRACKER = "KCF Tracker"

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
    cv2.imshow(WINDOW_TRACKER, frame)

    # Set fixed position for the tracking window (executed once)
    if len(gif_frames_kcf) == 1:
        cv2.moveWindow(WINDOW_TRACKER, WINDOW_START_X, WINDOW_START_Y)

    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Зберігаємо результат у GIF
if gif_frames_kcf:
    gif_filename_kcf = 'kcf_tracking.gif'
    output_path_kcf = os.path.join(output_folder, gif_filename_kcf)
    
    print(f"Creating KCF tracking GIF with {len(gif_frames_kcf)} frames...")
    # NOTE: 'create_gif' function is assumed to be defined elsewhere
    create_gif(gif_frames_kcf, output_path_kcf, fps=10) 

# 10. Звільняємо ресурси
video_kcf.release()
cv2.destroyAllWindows()
print("KCF tracking finished.") 
'''
'''
#------------------------------------------------------------------------#
#-----------------------------2 CSRT-------------------------------------#
#------------------------------------------------------------------------#
print("\nStarting 2: CSRT Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# --- CSRT CONFIGURATION PARAMETERS ---
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
WINDOW_ROI = "Select Bird to Track"
cv2.namedWindow(WINDOW_ROI)
# Set fixed position for the ROI selection window
cv2.moveWindow(WINDOW_ROI, WINDOW_START_X, WINDOW_START_Y) 

print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)
cv2.destroyWindow(WINDOW_ROI)

# 5. Initialize the tracker with the first frame and the selected box
tracker.init(first_frame, bbox)

# 6. Prepare for the loop
gif_frames_csrt = []
WINDOW_TRACKER = "CSRT Tracker"

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
    
    # Show the real-time result (опційно)
    cv2.imshow(WINDOW_TRACKER, frame)

    # Set fixed position for the tracking window (executed once)
    if len(gif_frames_csrt) == 1:
        cv2.moveWindow(WINDOW_TRACKER, WINDOW_START_X, WINDOW_START_Y)
    
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

#------------------------------------------------------------------------#
#-----------------------------Part 3: MIL--------------------------------#
#------------------------------------------------------------------------#
print("\nStarting Part 3: MIL Tracking...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")

# 1. Ініціалізація
video_mil = VIDEO
# Використовуємо константу OUTPUT_FOLDER для збереження всіх результатів
output_folder = OUTPUT_FOLDER
os.makedirs(output_folder, exist_ok=True)

# 2. Створюємо об'єкт трекера MIL
try:
    # Спробуємо стандартний метод, а потім legacy, як запасний варіант
    tracker = cv2.TrackerMIL_create()
except AttributeError:
    try:
        # Використовуємо cv2.legacy для доступу до старих трекерів (якщо потрібно)
        tracker = cv2.legacy.TrackerMIL_create()
    except Exception as e:
        print(f"ERROR: Failed to create MIL tracker. Ensure opencv-contrib-python is installed. Error: {e}")
        tracker = None

if tracker is None:
    exit()


# 3. Зчитуємо перший кадр
ret, first_frame = video_mil.read()
if not ret:
    print("Failed to read video")
    exit()

# 4. Дозволяємо користувачу вибрати об'єкт для відстеження
WINDOW_ROI = "Select Bird to Track"
cv2.namedWindow(WINDOW_ROI)
# Встановлюємо фіксовану позицію для вікна вибору
cv2.moveWindow(WINDOW_ROI, WINDOW_START_X, WINDOW_START_Y) 

print("Please select a bounding box for the bird and press ENTER or SPACE.")
bbox = cv2.selectROI(WINDOW_ROI, first_frame, False)
cv2.destroyWindow(WINDOW_ROI)

# 5. Ініціалізуємо трекер першим кадром та вибраною рамкою
tracker.init(first_frame, bbox)

# 6. Готуємося до циклу
gif_frames_mil = []
WINDOW_TRACKER = "MIL Tracker"

while True:
    ret, frame = video_mil.read()
    if not ret:
        print("End of video for MIL tracking.")
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
    gif_frames_mil.append(frame)
    
    # Показуємо результат в реальному часі (опційно)
    cv2.imshow(WINDOW_TRACKER, frame)

    # Встановлюємо фіксовану позицію для вікна трекінгу (виконується один раз)
    if len(gif_frames_mil) == 1:
        cv2.moveWindow(WINDOW_TRACKER, WINDOW_START_X, WINDOW_START_Y)

    # Вихід по натисканню 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Зберігаємо результат у GIF
if gif_frames_mil:
    gif_filename_mil = 'mil_tracking.gif'
    output_path_mil = os.path.join(output_folder, gif_filename_mil)
    
    print(f"Creating MIL tracking GIF with {len(gif_frames_mil)} frames...")
    # NOTE: 'create_gif' function is assumed to be defined elsewhere
    create_gif(gif_frames_mil, output_path_mil, fps=10) 

# 10. Звільняємо ресурси
video_mil.release()
cv2.destroyAllWindows()
print("MIL tracking finished.")