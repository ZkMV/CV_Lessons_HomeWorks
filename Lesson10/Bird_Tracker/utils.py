# utils.py
import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# ---- Шляхи до директорій проєкту ----
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "source"
DATA_DIR = BASE_DIR / "data"
RES_DIR  = BASE_DIR / "result"

def ensure_dirs(*dirs):
    """Створює директорії, якщо вони не існують."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

# ---- Функції для роботи з JSON файлами ----
def load_json(path):
    """Завантажує дані з JSON файлу."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    """Зберігає об'єкт у JSON файл."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---- Прогрес-бар для ітерацій ----
try:
    from tqdm import tqdm as _tqdm
    def progress_iter(iterable, total=None, desc=""):
        return _tqdm(iterable, total=total, desc=desc, unit="f",
                     ascii=True, dynamic_ncols=True, smoothing=0, leave=True)
except ImportError:
    _tqdm = None
    class _MiniBar:
        def __init__(self, total, desc=""):
            self.t = total if total is not None else 0
            self.i = 0
            self.desc = desc
            self.last_update_time = time.time()
        def update(self, n=1):
            self.i += n
            now = time.time()
            if now - self.last_update_time >= 0.1 or self.i == self.t:
                pct = (100 * self.i / self.t) if self.t else 0
                bar = "#" * int(pct / 2)
                sys.stdout.write(f"\r{self.desc} [{bar:<50}] {pct:5.1f}%")
                sys.stdout.flush()
                self.last_update_time = now
        def close(self):
            sys.stdout.write("\n")
            sys.stdout.flush()
    def progress_iter(iterable, total=None, desc=""):
        bar = _MiniBar(total if total is not None else (len(iterable) if hasattr(iterable, '__len__') else 0), desc)
        for x in iterable:
            yield x
            bar.update(1)
        bar.close()

# ---- Функції для роботи з відео ----
def write_video(frames, out_path, size, fps, desc="Writing video"):
    """Записує послідовність кадрів у відеофайл."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(size[0]), int(size[1])))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")
    
    total = len(frames) if isinstance(frames, (list, tuple)) else None
    
    for fr in progress_iter(frames, total=total, desc=desc):
        if fr.shape[1] != size[0] or fr.shape[0] != size[1]:
            fr = cv2.resize(fr, (int(size[0]), int(size[1])))
        vw.write(fr)
    vw.release()

def read_specific_frame(video_path, frame_number):
    """Читає та повертає конкретний кадр з відеофайлу за його номером."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total_frames:
        print(f"Error: Frame number {frame_number} is out of bounds. Video has {total_frames} frames.")
        cap.release()
        return None
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Failed to read frame {frame_number}.")
        return None
        
    return frame

# ---- Функції для роботи з UI та зображеннями ----
def select_roi_ui(frame, window_title="Select ROI"):
    """
    Показує вікно, в якому користувач може виділити прямокутну область (ROI).
    Приймає кадр для відображення та заголовок вікна.
    """
    H, W = frame.shape[:2]
    win_w = min(1280, max(480, W))
    win_h = min(720, max(360, H))
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_title, 50, 50)
    cv2.resizeWindow(window_title, win_w, win_h)
    cv2.imshow(window_title, frame)
    
    print("Select a bounding box and press ENTER or SPACE...")
    print("Cancel the selection process by pressing 'c' button!")
    
    roi = cv2.selectROI(window_title, frame, False, False)
    cv2.destroyWindow(window_title)
    
    if roi is None or roi[2] < 2 or roi[3] < 2:
        return None
    
    return tuple(map(int, roi))

# +++ ДОДАНО НОВУ ФУНКЦІЮ (ПЕРЕНЕСЕНО З step01) +++
def draw_label(img, text, org, scale=0.7, color=(255,255,255), thickness=1):
    """Малює текст з обводкою на зображенні."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Обводка
    cv2.putText(img, text, org, font, scale, (0,0,0), thickness + 2, cv2.LINE_AA)
    # Основний текст
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)