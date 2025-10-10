# step01_motion.py
from pathlib import Path
import cv2
import numpy as np
# +++ Змінено імпорт: додано draw_label +++
from utils import progress_iter, save_json, write_video, DATA_DIR, draw_label

# --- Функцію _draw_label видалено звідси ---

def _estimate_global_affine(prev_gray, gray,
                            nfeatures=1000, fast_thresh=10,
                            ransac_thr=3.0, max_iters=2000, conf=0.99):
    try:
        orb = cv2.ORB_create(nfeatures=int(nfeatures), fastThreshold=int(fast_thresh))
        k1, d1 = orb.detectAndCompute(prev_gray, None)
        k2, d2 = orb.detectAndCompute(gray, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return np.array([[1,0,0],[0,1,0]], np.float32), 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        m  = bf.match(d1, d2)
        if len(m) < 8:
            return np.array([[1,0,0],[0,1,0]], np.float32), 0

        m    = sorted(m, key=lambda x: x.distance)[:200]
        pts1 = np.float32([k1[a.queryIdx].pt for a in m])
        pts2 = np.float32([k2[a.trainIdx].pt  for a in m])

        A, inl = cv2.estimateAffinePartial2D(
            pts1, pts2, method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thr),
            maxIters=int(max_iters), confidence=float(conf)
        )
        if A is None:
            return np.array([[1,0,0],[0,1,0]], np.float32), 0
        n_inl = int(np.sum(inl)) if inl is not None else 0
        return A.astype(np.float32), n_inl
    except Exception:
        return np.array([[1,0,0],[0,1,0]], np.float32), 0

def _apply_pre_smooth(gray, ps_cfg):
    if gray is None or gray.ndim != 2:
        return gray
    t = (ps_cfg.get("type") or "none").lower()
    if t == "none":
        return gray
    if t == "gaussian":
        k = int(ps_cfg.get("ksize", 5)) | 1
        sigmaX = float(ps_cfg.get("sigmaX", 1.0))
        return cv2.GaussianBlur(gray, (k, k), sigmaX)
    if t == "bilateral":
        sig_c = float(ps_cfg.get("bilateral_sigma_color", 20))
        sig_s = float(ps_cfg.get("bilateral_sigma_space", 6))
        return cv2.bilateralFilter(gray, d=0, sigmaColor=sig_c, sigmaSpace=sig_s)
    if t == "median":
        k = int(ps_cfg.get("ksize", 5)) | 1
        return cv2.medianBlur(gray, k)
    return gray

def run(BASE, cfg, out_video_path, out_transforms_json):
    src_path = BASE / cfg["source_video"]
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_path}")

    fps = cfg["io"]["fps"] or cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dbg = cfg.get("debug", {}) or {}
    frac = float(dbg.get("process_fraction", 1.0))
    frac = max(0.0, min(1.0, frac))
    N = max(1, int(round(N_total * frac)))

    motion_cfg = cfg.get("motion", {})
    fb = motion_cfg.get("farneback", {})
    pyr_scale = float(fb.get("pyr_scale", 0.5))
    levels    = int(fb.get("levels", 3))
    winsize   = int(fb.get("winsize", 15))
    iterations= int(fb.get("iterations", 3))
    poly_n    = int(fb.get("poly_n", 5))
    poly_sigma= float(fb.get("poly_sigma", 1.2))

    pre_smooth_cfg = motion_cfg.get("pre_smooth", {})

    nfeatures   = int(motion_cfg.get("max_features", 1000))
    fast_thresh = int(motion_cfg.get("fast_threshold", 10))
    ransac_thr  = float(motion_cfg.get("ransac_reproj_thresh", 3.0))
    max_iters   = int(motion_cfg.get("ransac_max_iters", 2000))
    conf        = float(motion_cfg.get("ransac_confidence", 0.99))
    gmc_min_inliers = int(motion_cfg.get("gmc_min_inliers", 20))

    thermo_frames_dir = DATA_DIR / "thermo_frames"
    thermo_frames_dir.mkdir(parents=True, exist_ok=True)

    ok, prev_color = cap.read()
    if not ok or prev_color is None:
        cap.release(); raise RuntimeError("Cannot read first frame.")
    prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

    frames_out = []
    blank_frame = np.zeros_like(prev_color)
    draw_label(blank_frame, "Residual flow (TR)", (12, 44), 1.0)
    
    frames_out.append(blank_frame)
    transforms = [np.eye(3, dtype=np.float64).tolist()]
    
    cv2.imwrite(str(thermo_frames_dir / "000.jpg"), frames_out[0])
    
    steps = max(0, N - 1)
    for frame_idx in progress_iter(range(1, N), total=steps, desc="Step1 Thermography (Turbo)"):
        ok, curr_color = cap.read()
        if not ok or curr_color is None:
            break
        gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)

        A, n_inliers = _estimate_global_affine(prev_gray, gray,
                                       nfeatures=nfeatures, fast_thresh=fast_thresh,
                                       ransac_thr=ransac_thr, max_iters=max_iters, conf=conf)

        if n_inliers < gmc_min_inliers:
            thermo = blank_frame.copy()
            draw_label(thermo, f"GMC Failed ({n_inliers})", (12, 80), 0.8, (0, 0, 255))
        else:
            warped_prev = cv2.warpAffine(prev_color, A, (W, H), flags=cv2.INTER_LINEAR)
            prev_warp_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
            prev_warp_gray_s = _apply_pre_smooth(prev_warp_gray, pre_smooth_cfg)
            gray_s = _apply_pre_smooth(gray, pre_smooth_cfg)
            flow = cv2.calcOpticalFlowFarneback(
                prev_warp_gray_s, gray_s, None,
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0
            )
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
            mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            thermo = cv2.applyColorMap(mag_vis, 20)
            # +++ Використовуємо імпортовану функцію +++
            draw_label(thermo, "Residual flow (TR)", (12, 44), 1.0)
        
        filename = f"{frame_idx:03d}.jpg"
        filepath = thermo_frames_dir / filename
        cv2.imwrite(str(filepath), thermo)
        
        frames_out.append(thermo)
        transforms.append(np.eye(3, dtype=np.float64).tolist())

        prev_color = curr_color
        prev_gray  = gray

    cap.release()
    write_video(frames_out, out_video_path, (W, H), fps, desc="Write 1.thermo")
    save_json(out_transforms_json, {"C_to_ref": transforms, "size": [W, H], "fps": float(fps), "processed_frames": len(frames_out)})