import sys
import cv2
from pathlib import Path
from utils import DATA_DIR, RES_DIR, ensure_dirs, load_json, save_json, read_specific_frame, select_roi_ui

# Визначаємо BASE_DIR тут, щоб уникнути циклічного імпорту
BASE_DIR = Path(__file__).resolve().parent

# --- DIAGNOSTIC INFORMATION ---
print("--- DIAGNOSTIC INFORMATION ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"OpenCV File Path: {cv2.__file__}")
print("----------------------------\n")
# --- END DIAGNOSTIC INFORMATION ---

def main():
    ensure_dirs(DATA_DIR, RES_DIR)
    cfg = load_json(BASE_DIR / "config.json")
    source_video_path = BASE_DIR / cfg["source_video"]

    # 0) ROI
    roi_json = DATA_DIR / "00_roi.json"
    if not roi_json.exists():
        start_frame_idx = cfg.get("start_frame", 0)
        frame_for_roi = read_specific_frame(source_video_path, start_frame_idx)
        
        if frame_for_roi is None:
            raise RuntimeError(f"Could not read frame {start_frame_idx} to select ROI.")

        roi = select_roi_ui(frame_for_roi, window_title=f"Select ROI on frame #{start_frame_idx}")
        if roi is None:
            raise RuntimeError("ROI not selected.")
        
        # Зберігаємо ROI разом з номером кадру, на якому він був обраний
        save_json(roi_json, {"frame": start_frame_idx, "roi": roi})
        
        init_frame_idx = start_frame_idx
        init_roi = roi
    else:
        print("0) Using cached ROI from data/00_roi.json")
        roi_data = load_json(roi_json)
        init_frame_idx = roi_data.get("frame", 0)
        init_roi = roi_data["roi"]

    # 1) Thermography
    thermo_mp4 = DATA_DIR / "1.thermo.mp4"
    transforms_json = DATA_DIR / "1.transforms.json"
    if not (thermo_mp4.exists() and transforms_json.exists()):
        from step01_motion import run as step1
        step1(BASE_DIR, cfg, thermo_mp4, transforms_json)
    else:
        print(f"1) Skipping thermography, {thermo_mp4.name} already exists.")

    # 2) Classification
    cls_json = DATA_DIR / "2.classification.json"
    if not cls_json.exists():
        from step02_quality import run as step2
        # +++ ЗМІНА: Видалено аргумент init_roi +++
        step2(BASE_DIR, cfg, thermo_mp4, cls_json, preview=DATA_DIR / "2.classified.mp4")
    else:
        print(f"2) Skipping classification, {cls_json.name} already exists.")

    # 3) Interpolation
    restored_mp4 = DATA_DIR / "3.thermo_restored.mp4"
    if not restored_mp4.exists():
        from step03_interpolate import run as step3
        step3(BASE_DIR, cfg, thermo_mp4, cls_json, restored_mp4)
    else:
        print(f"3) Skipping interpolation, {restored_mp4.name} already exists.")

    # 4) Tracking (+ grayscale preview)
    track_json = DATA_DIR / "4.track.json"
    gray_preview = DATA_DIR / "4.gray-scale_track.mp4"
    if (not track_json.exists()) or (not gray_preview.exists()):
        from step04_track import run as step4
        # Передаємо ROI та індекс кадру, на якому він був визначений
        step4(BASE_DIR, cfg, restored_mp4, init_frame_idx, init_roi, cls_json, track_json, gray_preview)
    else:
        print(f"4) Skipping tracking, {track_json.name} already exists.")

    # 5) Overlay to original
    final_mp4 = RES_DIR / "overlay_tracked.mp4"
    if not final_mp4.exists():
        from step05_overlay import run as step5
        step5(BASE_DIR, cfg, source_video_path, track_json, transforms_json, final_mp4)
    else:
        print(f"5) Skipping final overlay, {final_mp4.name} already exists.")

if __name__ == "__main__":
    main()