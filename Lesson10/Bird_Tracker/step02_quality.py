import cv2
import json
import numpy as np
from scipy.stats import kurtosis
from utils import progress_iter, save_json, write_video, draw_label, DATA_DIR

def check_frame_quality(gray, active_thr, morph_ksize, min_kurtosis, max_contours, min_area):
    """
    Комплексна перевірка якості кадру.
    Повертає: (is_good, {kurtosis, num_contours, max_area})
    """
    if not gray.size:
        return False, {"kurtosis": -10, "contours": 999, "max_area": 0}

    # --- Перевірка 1: Ексцес (Kurtosis) ---
    active_pixels = gray[gray > active_thr]
    if active_pixels.size < 50:
        return False, {"kurtosis": -10, "contours": 0, "max_area": 0}

    k = kurtosis(active_pixels, fisher=True, bias=False)
    if k < min_kurtosis:
        return False, {"kurtosis": k, "contours": -1, "max_area": -1}

    # --- Перевірка 2: Контури (кількість і площа) ---
    _, thresh = cv2.threshold(gray, int(active_thr), 255, cv2.THRESH_BINARY)
    
    if morph_ksize > 0:
        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_noise_area = 5
    contours = [c for c in contours if cv2.contourArea(c) > min_noise_area]

    num_contours = len(contours)
    if num_contours == 0:
        return False, {"kurtosis": k, "contours": 0, "max_area": 0}
    
    largest_area = max(cv2.contourArea(c) for c in contours)

    # Фінальне рішення: кадр "хороший", якщо пройшов всі перевірки
    is_good = (num_contours <= max_contours) and (largest_area >= min_area)
    
    metrics = {"kurtosis": k, "contours": num_contours, "max_area": largest_area}
    return is_good, metrics

def run(BASE, cfg, thermo_mp4_path, out_json_path, preview=None):
    cap = cv2.VideoCapture(str(thermo_mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {thermo_mp4_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion_cfg = cfg.get("motion", {}) or {}
    class_cfg = cfg.get("classification", {}) or {}
    
    crop_pixels = int(motion_cfg.get("crop_pixels", 0))
    metric = class_cfg.get("metric")
    act_th = int(class_cfg.get("active_threshold", 35))
    morph_ksize = int(class_cfg.get("morph_open_ksize", 0))

    params = class_cfg.get("params", {})
    min_kurtosis = float(params.get("min_kurtosis", 3.0))
    max_contours = int(params.get("max_contours", 30))
    min_area = int(params.get("min_area", 50))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    if preview:
        quality_frames_dir = DATA_DIR / "quality_frames"
        quality_frames_dir.mkdir(parents=True, exist_ok=True)

    labels, values = [], []
    frames_vis = []

    for i in progress_iter(range(N), total=N, desc="Step2 Quality"):
        ok, fr = cap.read()
        if not ok: break

        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        if crop_pixels > 0:
            frame_h, frame_w = g.shape
            g_cropped = g[crop_pixels : frame_h - crop_pixels, crop_pixels : frame_w - crop_pixels]
        else:
            g_cropped = g
        
        g_enhanced = clahe.apply(g_cropped)
        
        good, metrics_values = check_frame_quality(g_enhanced, act_th, morph_ksize, min_kurtosis, max_contours, min_area)
        
        labels.append("good" if good else "bad")
        values.append(metrics_values)

        if preview:
            # +++ ЗМІНЕНО: Повертаємо візуалізацію покращеного сірого кадру замість бінарної маски +++
            vis_enhanced_bgr = cv2.cvtColor(g_enhanced, cv2.COLOR_GRAY2BGR)
            
            vis = np.zeros_like(fr)
            if crop_pixels > 0:
                vis[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels] = vis_enhanced_bgr
            else:
                vis = vis_enhanced_bgr

            color = (0, 255, 0) if good else (0, 0, 255)
            if crop_pixels > 0:
                cv2.rectangle(vis, (0, 0), (W, crop_pixels), color, -1)
                cv2.rectangle(vis, (0, H - crop_pixels), (W, H), color, -1)
                cv2.rectangle(vis, (0, 0), (crop_pixels, H), color, -1)
                cv2.rectangle(vis, (W - crop_pixels, 0), (W, H), color, -1)
            
            kurt_val = metrics_values['kurtosis']
            cnt_val = metrics_values['contours']
            area_val = metrics_values['max_area']
            
            # Текст на кадрі тепер показує всі три метрики, які використовуються для прийняття рішення
            txt = f"{'GOOD' if good else 'BAD'} | K:{kurt_val:.1f} C:{cnt_val} A:{int(area_val)}"
            draw_label(vis, txt, (20, 40), scale=0.6, color=(255,255,255), thickness=1)
            
            frames_vis.append(vis)
            
            filename = f"{i:03d}.jpg"
            filepath = quality_frames_dir / filename
            cv2.imwrite(str(filepath), vis)

    cap.release()
    save_json(out_json_path, {
        "labels": labels,
        "values": values,
        "metric": metric,
        "params": params
    })

    if preview:
        write_video(frames_vis, preview, (W, H), fps, desc="Write 2.classified")