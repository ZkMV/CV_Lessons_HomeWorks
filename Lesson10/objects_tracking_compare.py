# objects_tracking_Kalman_GMC.py
# Hybrid: CSRT + Kalman + GMC + optical-flow residual + anti-distractor + quad view MP4
# Ця версія НЕ визначає VIDEO/OUTPUT_FOLDER — очікує, що вони вже оголошені вище.

import os
import cv2
import numpy as np
from collections import deque

OUTPUT_FOLDER = os.path.join('Lesson10', 'result')
VIDEO = cv2.VideoCapture('Lesson10/source/hawk5.mp4')

# --------------------------- User parameters ------------------------------

# --- CSRT ---
USE_HOG_FEATURES = True
HISTOGRAM_BINS   = 32

# --- Kalman + Fusion ---
DT = 1.0
PREDICT_AHEAD_FRAMES  = 8
SIM_THRESHOLD_WEAK    = 0.35
SIM_THRESHOLD_GOOD    = 0.55
CONSISTENCY_MAX_PX    = 120
CONF_LOSS_THRESHOLD   = 0.55
CONF_STRONG_THRESHOLD = 0.75
CONF_BLEND_SIM        = 0.45

# --- Template Matching (reacquire) ---
TM_METHOD       = cv2.TM_CCOEFF_NORMED
TM_SEARCH_SCALE = 3.0
TM_MIN_SCORE    = 0.58

# --- Global Motion (ego-motion) ---
GMC_USE               = True
ORB_N_FEATURES        = 1200
ORB_FAST_THRESHOLD    = 12
GMC_MIN_INLIERS       = 25
GMC_RANSAC_REPROJ_THR = 2.5

# --- Optical flow (Farnebäck) ---
FLOW_PYR_SCALE  = 0.5
FLOW_LEVELS     = 3
FLOW_WINSIZE    = 21
FLOW_ITERS      = 3
FLOW_POLY_N     = 5
FLOW_POLY_SIGMA = 1.2

# --- Motion gating / anti-distractor ---
MOTION_NORM_PX        = 10.0
MOTION_MIN_FOR_ACCEPT = 0.25

LAB_L_MAX_Z   = 1.6
EDGE_DENS_MIN = 0.08
FG_RATIO_MIN  = 0.22

W_MOT  = 0.45
W_EDGE = 0.20
W_FG   = 0.20
W_L    = 0.35

# Напрямок руху (узгодженість із вектором Калмана)
W_DIR      = 0.25
DIR_MIN_OK = 0.55
BG_SIM_MAX = 0.68   # зарезервовано на майбутнє

AD_HARD_REJECT = True

# --- Drawing / windows ---
BOX_THICK = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_START_X = 50
WINDOW_START_Y = 50
ROI_WINDOW_W   = 1400
ROI_WINDOW_H   = 900
MAIN_WINDOW_W  = 1400
MAIN_WINDOW_H  = 840

# ------------------------------ Utils ------------------------------------

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
    if w <= 0 or h <= 0: return None
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

# ---- Anti-distractor helpers ----

def bbox_lab_stats(image_bgr, bbox):
    patch = extract_patch(image_bgr, bbox)
    if patch is None: return None, None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    L = lab[...,0].astype(np.float32)
    return float(np.mean(L)), float(np.std(L))

def edge_density(gray, bbox):
    patch = extract_patch(gray, bbox)
    if patch is None: return 0.0
    edges = cv2.Canny(patch, 50, 150, L2gradient=True)
    return float(np.count_nonzero(edges)) / float(edges.size + 1e-6)

def fg_ratio_from_mask(mask, bbox):
    x, y, w, h = clip_bbox_to_frame(bbox, mask.shape)
    roi = mask[y:y+h, x:x+w]
    if roi.size == 0: return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)

def flow_dir_consistency(flow, bbox, vx, vy):
    patch = extract_patch(flow, bbox)
    if patch is None: return 0.0
    v = np.array([vx, vy], dtype=np.float32)
    speed = np.linalg.norm(v) + 1e-6
    if speed < 0.2:
        return 0.5
    u = patch.reshape(-1,2)
    mag = np.linalg.norm(u,axis=1) + 1e-6
    dot = (u @ v) / (mag*speed)
    dot = np.clip(dot, -1.0, 1.0)
    good = dot[mag > np.median(mag)]
    if good.size == 0: return 0.0
    return float(np.mean((good + 1.0)/2.0))  # 0..1

# ---- Flow / GMC ----

def compute_flow(prev_gray, gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        pyr_scale=FLOW_PYR_SCALE,
                                        levels=FLOW_LEVELS,
                                        winsize=FLOW_WINSIZE,
                                        iterations=FLOW_ITERS,
                                        poly_n=FLOW_POLY_N,
                                        poly_sigma=FLOW_POLY_SIGMA,
                                        flags=0)

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
    cx = x + w/2.0; cy = y + h/2.0
    cx2, cy2 = warp_point_affine(cx, cy, A)
    a11, a12, _ = A[0]
    a21, a22, _ = A[1]
    sx = np.sqrt(a11*a11 + a21*a21);  sy = np.sqrt(a12*a12 + a22*a22)
    if sx <= 1e-6: sx = 1.0
    if sy <= 1e-6: sy = 1.0
    w2 = int(max(1, round(w * sx))); h2 = int(max(1, round(h * sy)))
    x2 = int(round(cx2 - w2/2.0));   y2 = int(round(cy2 - h2/2.0))
    return (x2, y2, w2, h2)

def affine_bg_flow(A, W, H):
    a11,a12,tx = A[0]; a21,a22,ty = A[1]
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X2 = a11*X + a12*Y + tx
    Y2 = a21*X + a22*Y + ty
    return np.dstack((X2 - X, Y2 - Y))

# ---- Motion / scoring helpers ----

def motion_score_from_mag(mag, bbox, norm_px=10.0):
    patch = extract_patch(mag, bbox)
    if patch is None: return 0.0
    v = float(np.mean(patch))
    return float(np.clip(v / max(1e-6, norm_px), 0.0, 1.0))

# -------------------------- Kalman construction --------------------------

def build_kalman(dt=1.0):
    # state: [cx, cy, vx, vy, w, h]^T ; meas: [cx, cy, w, h]^T
    kf = cv2.KalmanFilter(6, 4, 0, cv2.CV_32F)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1,  0, 0, 0],
        [0, 0, 0,  1, 0, 0],
        [0, 0, 0,  0, 1, 0],
        [0, 0, 0,  0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov     = np.diag([1e-2, 1e-2, 1e-1, 1e-1, 1e-3, 1e-3]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.5e-2, 2.5e-2, 1e-2, 1e-2]).astype(np.float32)
    kf.errorCovPost        = np.diag([1, 1, 10, 10, 1, 1]).astype(np.float32)
    return kf

# ----------------------------- Main logic -------------------------------

print("\nStarting Hybrid Tracking (CSRT + Kalman + Similarity + GMC + MOTION + QUAD)...")
print(f"Using OUTPUT_FOLDER = {os.path.abspath(OUTPUT_FOLDER)}")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# fps -> DT
_fps = VIDEO.get(cv2.CAP_PROP_FPS) if hasattr(VIDEO, "get") else 0.0
if _fps and _fps > 1e-3:
    DT = 1.0 / _fps
else:
    _fps = 25.0

video = VIDEO
ret, first_frame = video.read()
if not ret:
    print("Failed to read video")
    raise SystemExit

H0, W0 = first_frame.shape[:2]

# ROI window for selection
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.moveWindow("Select ROI", WINDOW_START_X, WINDOW_START_Y)
cv2.resizeWindow("Select ROI", min(ROI_WINDOW_W, W0), min(ROI_WINDOW_H, H0))
cv2.imshow("Select ROI", first_frame)
print("Please select a bounding box for the object and press ENTER or SPACE.")
init_bbox = cv2.selectROI("Select ROI", first_frame, False)
cv2.destroyWindow("Select ROI")
if init_bbox is None or init_bbox[2] <= 1 or init_bbox[3] <= 1:
    print("Invalid ROI")
    raise SystemExit

# CSRT
params = cv2.TrackerCSRT_Params()
params.use_hog        = USE_HOG_FEATURES
params.histogram_bins = HISTOGRAM_BINS
tracker = cv2.TrackerCSRT_create(params)
tracker.init(first_frame, init_bbox)

# Kalman
kf = build_kalman(DT)
cx, cy, w, h = bbox_to_cxcywh(init_bbox)
kf.statePost = np.array([[cx],[cy],[0.0],[0.0],[w],[h]], dtype=np.float32)

# Template / hist
template0 = extract_patch(first_frame, clip_bbox_to_frame(init_bbox, first_frame.shape))
hist0 = hsv_hist_corr(template0, bins=HISTOGRAM_BINS)

# Quad canvas + MP4 writer
tile_w, tile_h = W0 // 2, H0 // 2
quad_size = (W0, H0)
mp4_path = os.path.join(OUTPUT_FOLDER, "hybrid_quad.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(mp4_path, fourcc, _fps, (quad_size[0], quad_size[1]))

cv2.namedWindow("Tracker QUAD", cv2.WINDOW_NORMAL)
cv2.moveWindow("Tracker QUAD", WINDOW_START_X, WINDOW_START_Y + 60)
cv2.resizeWindow("Tracker QUAD", min(MAIN_WINDOW_W, quad_size[0]), min(MAIN_WINDOW_H, quad_size[1]))

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
acc_mask = np.zeros((H0, W0), np.float32)
ACC_ALPHA = 0.45
MAD_K     = 2.8
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

L0_mean, L0_std = bbox_lab_stats(first_frame, init_bbox)
if L0_std is None or L0_std < 1.0: L0_std = 1.0

def robust_motion_mask(res_mag):
    """Binary motion mask з MAD-порігом + експоненційною акумуляцією."""
    global acc_mask
    med = np.median(res_mag)
    mad = np.median(np.abs(res_mag - med)) + 1e-6
    thr = med + MAD_K * mad
    raw = (res_mag > thr).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, KERNEL, iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    acc_mask = ACC_ALPHA * raw.astype(np.float32) + (1.0 - ACC_ALPHA) * acc_mask
    return (acc_mask > 127).astype(np.uint8) * 255

def colorize_mag(mag):
    m = np.clip((mag / (np.percentile(mag, 99.0) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_TURBO)

conf_history = deque(maxlen=50)
pred_ahead_left = 0
frame_idx = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("End of video.")
        break

    frame_vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 0) GMC
    A = estimate_global_affine(prev_gray, gray) if GMC_USE else np.array([[1,0,0],[0,1,0]], dtype=np.float32)

    # 0a) Flow + residual
    flow   = compute_flow(prev_gray, gray)
    bgflow = affine_bg_flow(A, W0, H0)
    res_flow = flow - bgflow
    mag = np.sqrt(res_flow[...,0]**2 + res_flow[...,1]**2)

    # 0b) Motion mask (MAD + accumulation)
    motion_mask = robust_motion_mask(mag)

    # 0c) GMC visual check
    prev_gray_warp = cv2.warpAffine(prev_gray, A, (W0, H0), flags=cv2.INTER_LINEAR)
    gmc_diff = cv2.absdiff(gray, prev_gray_warp)

    # 1) Kalman predict (+ affine to current coords)
    kf.predict()
    pred_cx, pred_cy, pred_vx, pred_vy, pred_w, pred_h = kf.statePre.flatten()
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

        m_cx, m_cy, m_w, m_h = bbox_to_cxcywh(csrt_bbox)
        p_cx, p_cy, _, _     = bbox_to_cxcywh(pred_bbox_aff)
        cons = motion_consistency(p_cx, p_cy, m_cx, m_cy, norm_px=CONSISTENCY_MAX_PX)

        mot   = motion_score_from_mag(mag, csrt_bbox, norm_px=MOTION_NORM_PX)
        L_m, L_s = bbox_lab_stats(frame, csrt_bbox)
        edens = edge_density(gray, csrt_bbox)
        fgr   = fg_ratio_from_mask(motion_mask, csrt_bbox)
        zL    = abs((L_m - L0_mean) / max(1.0, L0_std)) if L_m is not None else 999.0
        dirc  = flow_dir_consistency(res_flow, csrt_bbox, pred_vx, pred_vy)

        conf_no_mot = compute_confidence(sim, cons, w_sim=CONF_BLEND_SIM)
        conf = float((1.0 - W_MOT)*conf_no_mot + W_MOT*mot + W_FG*fgr + W_EDGE*edens + W_DIR*dirc
                     - W_L*min(1.0, zL / LAB_L_MAX_Z))
        conf_history.append(conf)

        hard_reject = AD_HARD_REJECT and (
            (fgr < FG_RATIO_MIN) or (edens < EDGE_DENS_MIN) or
            (zL > LAB_L_MAX_Z)   or (dirc < DIR_MIN_OK)
        )

        if hard_reject or conf < CONF_LOSS_THRESHOLD or sim < SIM_THRESHOLD_WEAK or mot < MOTION_MIN_FOR_ACCEPT:
            reacq_bbox, tm_score = template_match_reacquire(frame, template0, pred_bbox_aff,
                                                            scale=TM_SEARCH_SCALE, method=TM_METHOD)
            if reacq_bbox is not None:
                mot_r     = motion_score_from_mag(mag, reacq_bbox, norm_px=MOTION_NORM_PX)
                Lr_m, Lr_s= bbox_lab_stats(frame, reacq_bbox)
                edens_r   = edge_density(gray, reacq_bbox)
                fgr_r     = fg_ratio_from_mask(motion_mask, reacq_bbox)
                zL_r      = abs((Lr_m - L0_mean) / max(1.0, L0_std)) if Lr_m is not None else 999.0
                dirc_r    = flow_dir_consistency(res_flow, reacq_bbox, pred_vx, pred_vy)
            else:
                mot_r = edens_r = fgr_r = dirc_r = 0.0
                zL_r = 999.0
                tm_score = 0.0

            accept_reacq = (reacq_bbox is not None and tm_score >= TM_MIN_SCORE and
                            mot_r >= MOTION_MIN_FOR_ACCEPT and fgr_r >= FG_RATIO_MIN and
                            edens_r >= EDGE_DENS_MIN and zL_r <= LAB_L_MAX_Z and
                            dirc_r >= DIR_MIN_OK)

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
                cv2.putText(frame_vis,
                            f"Reacq TM:{tm_score:.2f} MOT:{mot_r:.2f} DIR:{dirc_r:.2f}",
                            (20, 70), FONT, 0.60, (0,255,255), 2, cv2.LINE_AA)
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
                    f"SIM:{sim:.2f} CONS:{cons:.2f} MOT:{mot:.2f} DIR:{dirc:.2f} CONF:{conf:.2f}",
                    (20, 40), FONT, 0.60, (255,255,255), 2, cv2.LINE_AA)

        # обережне оновлення шаблону
        if conf >= CONF_STRONG_THRESHOLD and mot >= MOTION_MIN_FOR_ACCEPT and \
           edens >= EDGE_DENS_MIN and fgr >= FG_RATIO_MIN and zL <= LAB_L_MAX_Z and dirc >= DIR_MIN_OK:
            template0 = extract_patch(frame, final_bbox)
            hist0 = hsv_hist_corr(template0, bins=HISTOGRAM_BINS)
            if L_m is not None:
                L0_mean = 0.9*L0_mean + 0.1*L_m
                L0_std  = max(1.0, 0.9*L0_std + 0.1*(L_s if L_s is not None else L0_std))

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

    # ---------- QUAD ----------
    panel_tl = cv2.resize(frame_vis, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    cv2.putText(panel_tl, "Tracking (TL)", (10, 22), FONT, 0.7, (0,255,255), 2, cv2.LINE_AA)

    heat    = colorize_mag(mag)
    panel_tr = cv2.resize(heat, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    cv2.putText(panel_tr, "Residual flow (TR)", (10, 22), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)

    mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    panel_bl = cv2.resize(mask_bgr, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel_bl, "Motion mask (BL)", (10, 22), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)

    diff_bgr = cv2.cvtColor(gmc_diff, cv2.COLOR_GRAY2BGR)
    panel_br = cv2.resize(diff_bgr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
    cv2.putText(panel_br, "GMC check |curr - warp(prev)| (BR)", (10, 22), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

    top    = np.hstack([panel_tl, panel_tr])
    bottom = np.hstack([panel_bl, panel_br])
    quad   = np.vstack([top, bottom])

    show_quad = cv2.resize(quad,
                           (min(MAIN_WINDOW_W, quad_size[0]), min(MAIN_WINDOW_H, quad_size[1])),
                           interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Tracker QUAD", show_quad)
    writer.write(quad)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    frame_idx += 1
    prev_gray = gray

writer.release()
video.release()
cv2.destroyAllWindows()
print(f"Hybrid tracking finished. MP4 saved to: {os.path.abspath(os.path.join(OUTPUT_FOLDER, 'hybrid_quad.mp4'))}")
