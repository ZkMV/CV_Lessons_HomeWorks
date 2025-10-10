import cv2
import json
import numpy as np
from utils import progress_iter, save_json, write_video

# --- Функції _create_tracker, _create_fallback_chain, _clamp_roi, _try_init_gray, _update_gray залишаються без змін ---
def _create_tracker(name="CSRT"):
    name = (name or "CSRT").upper()
    def _mk(csrt=True):
        if csrt:
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
        else:
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()
        return None
    if name == "KCF":
        return _mk(csrt=False) or _mk(csrt=True)
    return _mk(csrt=True) or _mk(csrt=False)

def _create_fallback_chain(preferred):
    chain = [preferred, ("KCF" if preferred != "KCF" else "CSRT"), "MOSSE"]
    out = []
    for name in chain:
        if name == "MOSSE":
            t = None
            if hasattr(cv2, "TrackerMOSSE_create"): t = cv2.TrackerMOSSE_create()
            elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
                t = cv2.legacy.TrackerMOSSE_create()
            if t is not None: out.append(("MOSSE", t))
        else:
            tr = _create_tracker(name)
            if tr is not None: out.append((name, tr))
    return out

def _clamp_roi(b, W, H):
    x, y, w, h = map(int, b)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)

def _try_init_gray(tracker, frame_bgr, roi):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    try:
        if tracker.init(g, tuple(map(float, roi))):
            return True, True
    except Exception:
        pass
    try:
        ok = tracker.init(frame_bgr, tuple(map(float, roi)))
        return ok, False
    except Exception:
        return False, False

def _update_gray(tracker, frame_bgr, use_gray):
    if use_gray:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return tracker.update(g)
    return tracker.update(frame_bgr)
    
# --- ОНОВЛЕНА ФУНКЦІЯ run ---
def run(BASE, cfg, restored_mp4_path, init_frame_idx, init_roi, classif_json_path, out_track_json, out_preview_mp4=None):
    cap = cv2.VideoCapture(str(restored_mp4_path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    motion_cfg = cfg.get("motion", {}) or {}
    crop_pixels = int(motion_cfg.get("crop_pixels", 0))

    # Змінюємо ROI відповідно до обрізки
    roi0 = (
        init_roi[0] - crop_pixels,
        init_roi[1] - crop_pixels,
        init_roi[2],
        init_roi[3]
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
    ok, fr0 = cap.read()
    if not ok or fr0 is None:
        cap.release()
        raise RuntimeError(f"Cannot read frame {init_frame_idx}.")

    # Обрізаємо кадр для трекера
    if crop_pixels > 0:
        fr0_cropped = fr0[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
        W_cropped, H_cropped = fr0_cropped.shape[1], fr0_cropped.shape[0]
    else:
        fr0_cropped = fr0
        W_cropped, H_cropped = W, H
        
    roi0 = _clamp_roi(roi0, W_cropped, H_cropped)

    preferred = (cfg.get("tracking", {}).get("tracker") or "CSRT").upper()
    chain = _create_fallback_chain(preferred)

    tracker = None
    used_name = None
    use_gray = True
    for name, t in chain:
        ok_init, used_gray = _try_init_gray(t, fr0_cropped, roi0)
        if ok_init:
            tracker = t
            used_name = name
            use_gray = used_gray
            break
    if tracker is None:
        cap.release()
        raise RuntimeError(f"Tracker init failed at {init_frame_idx}. ROI={roi0}. Tried: {', '.join(n for n,_ in chain)}")

    tracks = [{"frame": init_frame_idx, "x": roi0[0] + crop_pixels, "y": roi0[1] + crop_pixels, "w": roi0[2], "h": roi0[3]}]

    # Форвард-трекінг
    for t in progress_iter(range(init_frame_idx + 1, n), total=n - (init_frame_idx + 1), desc=f"Step4 Tracking [{used_name}|{'GRAY' if use_gray else 'BGR'}]"):
        ok, fr = cap.read()
        if not ok: break

        # Обрізаємо кадр
        if crop_pixels > 0:
            fr_cropped = fr[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
        else:
            fr_cropped = fr

        ok2, bbox = _update_gray(tracker, fr_cropped, use_gray)
        if not ok2:
            last = tracks[-1]
            # Координати останнього bbox вже в повній системі, тому треба їх перерахувати для обрізаного кадру
            last_cropped_x = last["x"] - crop_pixels
            last_cropped_y = last["y"] - crop_pixels
            bbox = (last_cropped_x, last_cropped_y, last["w"], last["h"])
        
        x, y, w, h = map(int, bbox)
        x, y, w, h = _clamp_roi((x, y, w, h), W_cropped, H_cropped)
        # Зберігаємо координати в системі повного кадру
        tracks.append({"frame": t, "x": x + crop_pixels, "y": y + crop_pixels, "w": w, "h": h})

    # Бекфіл (зворотній трекінг)
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
    ok, fr0b = cap.read()
    if ok and fr0b is not None:
        if crop_pixels > 0:
            fr0b_cropped = fr0b[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
        else:
            fr0b_cropped = fr0b
        
        if used_name == "MOSSE":
            t2 = cv2.TrackerMOSSE_create() if hasattr(cv2, "TrackerMOSSE_create") else cv2.legacy.TrackerMOSSE_create()
        else:
            t2 = _create_tracker(used_name)

        back = []
        if t2 is not None:
            ok_init2, used_gray_b = _try_init_gray(t2, fr0b_cropped, roi0)
            if ok_init2:
                for ti in progress_iter(range(init_frame_idx - 1, -1, -1), total=init_frame_idx, desc=f"Step4 Backfill [{used_name}|{'GRAY' if used_gray_b else 'BGR'}]"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, ti)
                    ok, frb = cap.read()
                    if not ok: break
                    
                    if crop_pixels > 0:
                        frb_cropped = frb[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
                    else:
                        frb_cropped = frb

                    ok2, bb2 = _update_gray(t2, frb_cropped, used_gray_b)
                    
                    if not ok2:
                        last_b = back[-1] if back else tracks[0]
                        last_cropped_x = last_b["x"] - crop_pixels
                        last_cropped_y = last_b["y"] - crop_pixels
                        bb2 = (last_cropped_x, last_cropped_y, last_b["w"], last_b["h"])
                        
                    x, y, w, h = map(int, bb2)
                    x, y, w, h = _clamp_roi((x, y, w, h), W_cropped, H_cropped)
                    back.append({"frame": ti, "x": x + crop_pixels, "y": y + crop_pixels, "w": w, "h": h})
        
        back.sort(key=lambda d: d["frame"])
        tracks = back + tracks

    cap.release()
    save_json(out_track_json, {"fps": fps, "tracker": used_name, "mode": ("GRAY" if use_gray else "BGR"), "tracks": tracks})

    # Створення прев'ю-відео
    if out_preview_mp4:
        bbmap = {tr["frame"]: (tr["x"], tr["y"], tr["w"], tr["h"]) for tr in tracks}
        cap2 = cv2.VideoCapture(str(restored_mp4_path))
        total = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in progress_iter(range(total), total=total, desc="Step4 Preview (grayscale)"):
            ok, fr = cap2.read()
            if not ok: break
            
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

            # Малюємо рамку, що показує область трекінгу
            if crop_pixels > 0:
                cv2.rectangle(vis, (crop_pixels, crop_pixels), (W - crop_pixels, H - crop_pixels), (0, 0, 150), 1)

            if i in bbmap:
                x, y, w, h = bbmap[i]
                # Bbox вже в координатах повного кадру, тому малюємо його як є
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            frames.append(vis)
        cap2.release()
        write_video(frames, out_preview_mp4, (W, H), fps, desc="Write 4.gray-scale_track")