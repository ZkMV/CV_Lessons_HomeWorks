# objects_tracking_Kalman_GMC.py
# Kalman + Global Motion Compensation + residual-flow search
# Motion/Residual-focused gating to avoid sticking to static edges (branches)
# 4-panel visualization and MP4 writer

import os
import cv2
import numpy as np

# -------------------------- Required constants --------------------------
OUTPUT_FOLDER = os.path.join('Lesson10', 'result')
VIDEO = cv2.VideoCapture('Lesson10/source/hawk5.mp4')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------- UI / layout ---------------------------------
WINDOW_START_X = 50
WINDOW_START_Y = 50
PANEL_W = 960
PANEL_H = 540
COLS = 2
ROWS = 2
CANVAS_W = PANEL_W * COLS
CANVAS_H = PANEL_H * ROWS

FONT = cv2.FONT_HERSHEY_SIMPLEX

def put_text(img, text, org, scale=0.7, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, FONT, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color,     thick,   cv2.LINE_AA)

def draw_bbox(img, bbox, color, label=None):
    x,y,w,h = bbox
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2, cv2.LINE_AA)
    if label:
        put_text(img, label, (x, max(0, y-6)), 0.6, color, 2)

def stack2x2(tl, tr, bl, br):
    def fit(x):
        return cv2.resize(x, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    return np.vstack([np.hstack([fit(tl), fit(tr)]),
                      np.hstack([fit(bl), fit(br)])])

# -------------------------- Geometry helpers ----------------------------
def bbox_to_cxcywh(b):
    x,y,w,h = b
    return np.array([x + w/2.0, y + h/2.0, w, h], dtype=np.float32)

def cxcywh_to_bbox(c):
    cx,cy,w,h = c
    return (int(cx - w/2.0), int(cy - h/2.0), int(w), int(h))

def clip_bbox_to_frame(b, shape):
    H,W = shape[:2]
    x,y,w,h = b
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x,y,w,h)

def warp_point_affine(x, y, A):
    vec = np.array([x, y, 1.0], dtype=np.float32)
    res = A @ vec
    return float(res[0]), float(res[1])

def warp_bbox_affine(b, A):
    x,y,w,h = b
    cx,cy = x + w/2.0, y + h/2.0
    cx2,cy2 = warp_point_affine(cx,cy,A)
    a11,a12,_ = A[0]; a21,a22,_ = A[1]
    sx = float(np.hypot(a11,a21)); sy = float(np.hypot(a12,a22))
    sx = sx if sx>1e-6 else 1.0
    sy = sy if sy>1e-6 else 1.0
    w2 = int(max(1, round(w*sx)))
    h2 = int(max(1, round(h*sy)))
    x2 = int(round(cx2 - w2/2.0))
    y2 = int(round(cy2 - h2/2.0))
    return (x2,y2,w2,h2)

def iou(a, b):
    ax,ay,aw,ah = a
    bx,by,bw,bh = b
    ix1,iy1 = max(ax,bx), max(ay,by)
    ix2,iy2 = min(ax+aw,bx+bw), min(ay+ah,by+bh)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter<=0: return 0.0
    u = aw*ah + bw*bh - inter
    return inter/float(max(1,u))

# -------------------------- Global motion (GMC) -------------------------
ORB_N_FEATURES = 1000
ORB_FAST_THRESHOLD = 10
GMC_MIN_INLIERS = 20
GMC_RANSAC_REPROJ_THR = 3.0

def estimate_global_affine(prev_gray, gray):
    """return A (2x3) and n_inliers"""
    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    k1,d1 = orb.detectAndCompute(prev_gray, None)
    k2,d2 = orb.detectAndCompute(gray, None)
    if d1 is None or d2 is None or len(k1)<8 or len(k2)<8:
        return np.array([[1,0,0],[0,1,0]], np.float32), 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d1,d2)
    if len(m)<8:
        return np.array([[1,0,0],[0,1,0]], np.float32), 0
    m = sorted(m, key=lambda t: t.distance)[:200]
    pts1 = np.float32([k1[t.queryIdx].pt for t in m])
    pts2 = np.float32([k2[t.trainIdx].pt for t in m])
    A, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC,
        ransacReprojThreshold=GMC_RANSAC_REPROJ_THR,
        maxIters=2000, confidence=0.99
    )
    if A is None:
        return np.array([[1,0,0],[0,1,0]], np.float32), 0
    ninl = int(np.count_nonzero(inliers)) if inliers is not None else 0
    if ninl < GMC_MIN_INLIERS:
        return np.array([[1,0,0],[0,1,0]], np.float32), ninl
    return A.astype(np.float32), ninl

# -------------------------- Kalman --------------------------------------
def build_kalman(dt=1.0):
    # state [cx, cy, vx, vy, w, h]^T ; meas [cx, cy, w, h]^T
    kf = cv2.KalmanFilter(6,4,0, cv2.CV_32F)
    kf.transitionMatrix = np.array([
        [1,0,dt,0, 0,0],
        [0,1,0, dt,0,0],
        [0,0,1, 0, 0,0],
        [0,0,0, 1, 0,0],
        [0,0,0, 0, 1,0],
        [0,0,0, 0, 0,1]
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], np.float32)
    kf.processNoiseCov = np.diag([1e-2,1e-2, 1e-1,1e-1, 1e-3,1e-3]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([2.5e-2,2.5e-2, 1e-2,1e-2]).astype(np.float32)
    kf.errorCovPost = np.diag([1,1, 10,10, 1,1]).astype(np.float32)
    return kf

# -------------------------- Defaults / thresholds -----------------------
# Пошукове вікно
SEARCH_EXPAND_A = 2.2
SEARCH_EXPAND_B = 1.8
SEARCH_EXPAND_C = 1.3   # зменшено, щоб менше «зачіпати» далекі контрасти

# Параметри побудови маски руху у вікні
PERC_HIGH_A = 99.3
PERC_HIGH_B = 99.5
LOW_FRAC_A  = 0.70
LOW_FRAC_B  = 0.80

MORPH_OPEN_A, MORPH_CLOSE_A = 3, 7
MORPH_OPEN_B, MORPH_CLOSE_B = 3, 9

# Геометричні гейти
MIN_AREA_FRAC = 0.015
MAX_AREA_FRAC = 0.40
ASPECT_MIN    = 1.6
ASPECT_MAX    = 6.0
ECC_MIN_A     = 0.70
ECC_MIN_B     = 0.80

EDGE_LOW, EDGE_HIGH = 80, 170
EDGE_DENS_MIN_A = 0.10
EDGE_DENS_MIN_B = 0.12

# «Яскравість» для приглушення злипань на світлих фонах
L_BRIGHT_TH    = 85          # LAB L* для глобальної bright-mask
LAB_L_MAX_GATE = 72.0        # гейт яскравості у кандидата

# Інші пороги
PREDICT_AHEAD_FRAMES = 6
T_UPDATE_CONF = 0.70
T_MIN_SIZE    = 18

# Оцінка якості кадру (Q)
Q_GOOD  = 0.60
Q_BAD   = 0.45
E_MAX_W = 40.0    # нормалізація середнього залишку
PHI_MAX = 0.35    # нормалізація частки «перенасичення» у залишку
EDGE_NORM = 0.10  # нормалізація щільності ребер
RES_VIS_CLIP = 35 # кліп для BR-візуалізації

# Нові ключові пороги проти «гілок»
RES_RATIO_MIN   = 0.18  # мін. частка «яскравого» залишку в кандидаті
MU_MAG_MIN      = 0.45  # мін. середня величина потоку в кандидаті
VAR_MAG_MIN     = 0.20  # або мін. дисперсія потоку

def main():
    cap = VIDEO
    if not cap or not cap.isOpened():
        print("Failed to open VIDEO")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt  = 1.0 / max(1.0, fps)

    ok, first = cap.read()
    if not ok:
        print("Failed to read first frame"); return
    H,W = first.shape[:2]

    # ROI selection
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Select ROI", WINDOW_START_X, WINDOW_START_Y)
    cv2.resizeWindow("Select ROI", min(W, PANEL_W), min(H, PANEL_H))
    cv2.imshow("Select ROI", first)
    print("Select a bounding box and press ENTER/SPACE...")
    init_bbox = cv2.selectROI("Select ROI", first, False)
    cv2.destroyWindow("Select ROI")
    if init_bbox is None or init_bbox[2]<=1 or init_bbox[3]<=1:
        print("Invalid ROI"); return

    # Kalman init
    kf = build_kalman(dt)
    cx0,cy0,w0,h0 = bbox_to_cxcywh(init_bbox)
    kf.statePost = np.array([[cx0],[cy0],[0.0],[0.0],[w0],[h0]], np.float32)
    Q_base = kf.processNoiseCov.copy()
    predict_streak = 0

    # Tiny NCC template (із серединки bbox)
    fx,fy,fw,fh = init_bbox
    mx = fx + fw//10; my = fy + fh//10
    mw2 = max(12, int(fw*0.8)); mh2 = max(12, int(fh*0.8))
    mx = max(0, min(mx, W-mw2)); my = max(0, min(my, H-mh2))
    template_ncc = first[my:my+mh2, mx:mx+mw2].copy()

    prev_color = first.copy()
    prev_gray  = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    # Writer + window
    out_path = os.path.join(OUTPUT_FOLDER, "motion_kalman_gmc_4view.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (CANVAS_W, CANVAS_H))
    cv2.namedWindow("Tracking (4v)", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Tracking (4v)", WINDOW_START_X, WINDOW_START_Y)
    cv2.resizeWindow("Tracking (4v)", CANVAS_W, CANVAS_H)

    last_good_score = 0.0  # гістерезис у режимі B

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bright mask на повному кадрі
        lab_full = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L_full = lab_full[:,:,0]
        bright_mask_full = (L_full > L_BRIGHT_TH).astype(np.uint8)
        bright_mask_full = cv2.morphologyEx(
            bright_mask_full, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), 1
        )

        # GMC (prev->curr)
        A, n_inliers = estimate_global_affine(prev_gray, gray)
        warped_prev = cv2.warpAffine(prev_color, A, (W,H), flags=cv2.INTER_LINEAR)

        # Residual для BR-панелі (фіксований кліп)
        residual_color = cv2.absdiff(frame, warped_prev)
        residual_gray = cv2.cvtColor(residual_color, cv2.COLOR_BGR2GRAY)
        res_vis = np.clip(residual_gray, 0, RES_VIS_CLIP)
        res_vis = (res_vis * (255.0/RES_VIS_CLIP)).astype(np.uint8)
        br_panel = cv2.cvtColor(res_vis, cv2.COLOR_GRAY2BGR)

        # Residual flow (Farnebäck) між warped_prev і curr
        prev_warp_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_warp_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
        mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_vis = cv2.applyColorMap(mag_vis, cv2.COLORMAP_TURBO)

        # Kalman predict
        kf.predict()
        pcx,pcy,_,_,pw,ph = kf.statePre.flatten()
        pred_bbox = cxcywh_to_bbox((pcx,pcy,pw,ph))
        pred_bbox = clip_bbox_to_frame(pred_bbox, frame.shape)
        pred_bbox_aff = warp_bbox_affine(pred_bbox, A)
        pred_bbox_aff = clip_bbox_to_frame(pred_bbox_aff, frame.shape)

        # ---------------- Frame-quality Q (локально у вікні W) ----------------
        # первинне вікно W для оцінки
        search_expand = SEARCH_EXPAND_A
        cxp = pred_bbox_aff[0] + pred_bbox_aff[2]/2.0
        cyp = pred_bbox_aff[1] + pred_bbox_aff[3]/2.0
        x0 = max(0, int(cxp - (pred_bbox_aff[2]*search_expand)/2.0))
        y0 = max(0, int(cyp - (pred_bbox_aff[3]*search_expand)/2.0))
        ww = int(max(8, pred_bbox_aff[2]*search_expand))
        hh = int(max(8, pred_bbox_aff[3]*search_expand))
        if x0+ww > W: ww = W - x0
        if y0+hh > H: hh = H - y0

        # Метрики якості у W
        resW = residual_gray[y0:y0+hh, x0:x0+ww]
        E_res   = float(np.mean(resW)) if resW.size else 0.0
        p_hi_res= float(np.mean(resW > RES_VIS_CLIP)) if resW.size else 0.0
        edgesW  = cv2.Canny(frame[y0:y0+hh, x0:x0+ww], EDGE_LOW, EDGE_HIGH) if ww>2 and hh>2 else None
        edge_density_W = float(np.mean(edgesW>0)) if edgesW is not None else 0.0

        q_inl = float(np.clip(n_inliers/50.0, 0.0, 1.0))
        q_res = float(np.clip(1.0 - (E_res / E_MAX_W), 0.0, 1.0))
        q_hi  = float(np.clip(1.0 - (p_hi_res / PHI_MAX), 0.0, 1.0))
        q_edg = float(np.clip(edge_density_W / EDGE_NORM, 0.0, 1.0))
        Q = 0.35*q_inl + 0.25*q_res + 0.20*q_hi + 0.20*q_edg

        # Режим A/B/C
        if Q >= Q_GOOD:
            mode = 'A'
            search_expand = SEARCH_EXPAND_A
            PERC_HIGH = PERC_HIGH_A
            LOW_FRAC  = LOW_FRAC_A
            MORPH_OPEN, MORPH_CLOSE = MORPH_OPEN_A, MORPH_CLOSE_A
            ecc_min = ECC_MIN_A
            edge_min = EDGE_DENS_MIN_A
        elif Q < Q_BAD:
            mode = 'C'
            search_expand = SEARCH_EXPAND_C
            PERC_HIGH = PERC_HIGH_B
            LOW_FRAC  = LOW_FRAC_B
            MORPH_OPEN, MORPH_CLOSE = MORPH_OPEN_B, MORPH_CLOSE_B
            ecc_min = ECC_MIN_B
            edge_min = EDGE_DENS_MIN_B
        else:
            mode = 'B'
            search_expand = SEARCH_EXPAND_B
            PERC_HIGH = PERC_HIGH_B
            LOW_FRAC  = LOW_FRAC_B
            MORPH_OPEN, MORPH_CLOSE = MORPH_OPEN_B, MORPH_CLOSE_B
            ecc_min = ECC_MIN_B
            edge_min = EDGE_DENS_MIN_B

        # оновлюємо W під обраний режим
        x0 = max(0, int(cxp - (pred_bbox_aff[2]*search_expand)/2.0))
        y0 = max(0, int(cyp - (pred_bbox_aff[3]*search_expand)/2.0))
        ww = int(max(8, pred_bbox_aff[2]*search_expand))
        hh = int(max(8, pred_bbox_aff[3]*search_expand))
        if x0+ww > W: ww = W - x0
        if y0+hh > H: hh = H - y0

        # Ініціалізуємо маску руху на кожному кадрі
        motion_mask = None

        frame_vis = frame.copy()
        final_bbox = pred_bbox_aff
        meas_ok = False
        conf    = 0.0
        best_S  = -1.0

        # ---------------- Режим C → без вимірів, але маска нульова ----------------
        if mode == 'C':
            if predict_streak == 0:
                predict_streak = PREDICT_AHEAD_FRAMES
            predict_streak = max(0, predict_streak - 1)
            scale = min(1.0 + 0.25*(PREDICT_AHEAD_FRAMES - predict_streak), 3.0)
            kf.processNoiseCov = (Q_base * scale).astype(np.float32)
            # порожня маска під BL-панель
            motion_mask = np.zeros((hh, ww), dtype=np.uint8)

        else:
            # ---------------- Локальний пошук кандидатів ----------------
            mag_win = mag[y0:y0+hh, x0:x0+ww].copy() if mag.size else np.zeros((hh,ww), np.float32)

            # двопорогова маска (перцентилі)
            flat = mag_win.ravel()
            hi = float(np.percentile(flat, PERC_HIGH)) if flat.size else 0.0
            lo = float(hi * LOW_FRAC)
            m_lo = (mag_win >= lo).astype(np.uint8)

            # морфологія
            motion_mask = cv2.morphologyEx(
                m_lo, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_OPEN, MORPH_OPEN))
            )
            motion_mask = cv2.morphologyEx(
                motion_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(MORPH_CLOSE, MORPH_CLOSE))
            )
            # відсікти дуже світлі зони у W
            bm_local = bright_mask_full[y0:y0+hh, x0:x0+ww]
            motion_mask[bm_local>0] = 0

            # Masked-NCC: лише на рух∩залишок
            ncc_score = None
            if template_ncc is not None and template_ncc.size:
                th,tw = template_ncc.shape[:2]
                if hh>th and ww>tw:
                    try:
                        # приглушимо позамаскові пікселі в копії
                        fm = frame[y0:y0+hh, x0:x0+ww].copy()
                        res_local = (residual_gray[y0:y0+hh, x0:x0+ww] > RES_VIS_CLIP).astype(np.uint8)
                        mask_ncc = (motion_mask & res_local)
                        fm[mask_ncc==0] = 0
                        tmpl = template_ncc.copy()
                        tmpl = cv2.GaussianBlur(tmpl, (3,3), 0)
                        fm   = cv2.GaussianBlur(fm,   (3,3), 0)
                        ncc = cv2.matchTemplate(fm, tmpl, cv2.TM_CCOEFF_NORMED)
                        ncc_score = cv2.resize(ncc, (ww,hh), interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        ncc_score = None

            contours,_ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # гейти кандидатів
            area_min = MIN_AREA_FRAC * (pred_bbox_aff[2]*pred_bbox_aff[3])
            area_max = MAX_AREA_FRAC * (pred_bbox_aff[2]*pred_bbox_aff[3])
            L_local  = L_full[y0:y0+hh, x0:x0+ww]
            edges_local = cv2.Canny(frame[y0:y0+hh, x0:x0+ww], EDGE_LOW, EDGE_HIGH) if ww>2 and hh>2 else None

            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                if w*h<=0: continue
                area = w*h
                if area<area_min or area>area_max: continue
                asp = w/float(h+1e-6)
                if asp<ASPECT_MIN or asp>ASPECT_MAX: continue

                # яскравість усередині кандидата
                L_roi = L_local[y:y+h, x:x+w]
                if L_roi.size and float(np.mean(L_roi)) > LAB_L_MAX_GATE:
                    continue
                # щільність ребер
                ed_local = 1.0
                if edges_local is not None:
                    edges_roi = edges_local[y:y+h, x:x+w]
                    ed_local = float(np.mean(edges_roi > 0))
                    if ed_local < edge_min:
                        continue
                # ексцентриситет
                M = cv2.moments(c)
                ecc = 0.0
                if M['mu20']+M['mu02']>1e-6:
                    a = float(M['mu20']); d = float(M['mu02']); b = float(M['mu11'])
                    tr = a + d
                    disc = max(0.0, (tr*0.5)**2 - (a*d - b*b))
                    lam1 = tr*0.5 + np.sqrt(disc)
                    lam2 = tr*0.5 - np.sqrt(disc)
                    if lam1>1e-9:
                        ecc = 1.0 - lam2/lam1
                if ecc < ecc_min:
                    continue

                # абсолютні координати кандидата
                cand = (x0+x, y0+y, w, h)

                # --------- НОВІ ключові гейти проти «гілок» ----------
                # 1) Частка залишку в кандидата
                res_roi = residual_gray[cand[1]:cand[1]+h, cand[0]:cand[0]+w]
                res_ratio = float(np.mean(res_roi > RES_VIS_CLIP)) if res_roi.size else 0.0
                if res_ratio < RES_RATIO_MIN:
                    continue

                # 2) Внутрішній рух
                mag_roi = mag[cand[1]:cand[1]+h, cand[0]:cand[0]+w]
                mu_mag = float(np.mean(mag_roi)) if mag_roi.size else 0.0
                var_mag = float(np.var(mag_roi)) if mag_roi.size else 0.0
                if not (mu_mag >= MU_MAG_MIN or var_mag >= VAR_MAG_MIN):
                    continue
                # ----------------------------------------------------

                # скорингові компоненти
                cx_i = cand[0] + w/2.0
                cy_i = cand[1] + h/2.0
                # близькість до прогнозу
                dist2 = (cx_i-pcx)**2 + (cy_i-pcy)**2
                sigma = (0.35*max(pw,ph))**2 + 1e-6
                dist_term = float(np.exp(-dist2/(2*sigma)))
                # IoU
                iou_term = iou(pred_bbox_aff, cand)
                # узгодження напрямку
                vx,vy = float(kf.statePre[2,0]), float(kf.statePre[3,0])
                vel_norm = float(np.hypot(vx,vy)+1e-6)
                dir_agree = 0.5
                if vel_norm>1e-3:
                    ang_roi = ang[cand[1]:cand[1]+h, cand[0]:cand[0]+w] if ang.size else None
                    if ang_roi is not None and ang_roi.size:
                        med_ang = float(np.median(ang_roi))
                        ux,uy = float(np.cos(np.deg2rad(med_ang))), float(np.sin(np.deg2rad(med_ang)))
                        dir_agree = float(max(0.0, (ux*(vx/vel_norm) + uy*(vy/vel_norm))))
                # masked-NCC (якщо є)
                ncc_term = 0.0
                if ncc_score is not None:
                    sx,sy = max(0,x), max(0,y)
                    ex,ey = min(ww-1, x+w-1), min(hh-1, y+h-1)
                    if ex>sx and ey>sy:
                        ncc_term = float(np.max(ncc_score[sy:ey, sx:ex]))
                        ncc_term = float(np.clip(ncc_term, 0.0, 1.0))

                # НОВИЙ скоринг: більше «руху/залишку», менше геометрії
                S = (0.28*dist_term
                     + 0.12*iou_term
                     + 0.22*float(np.clip(mu_mag/3.0, 0.0, 1.0))
                     + 0.22*res_ratio
                     + 0.10*dir_agree
                     + 0.06*ncc_term)

                # штраф «фоноподібності»
                bg_penalty = float(np.clip(1.0 - res_ratio*1.6, 0.0, 1.0))
                S -= 0.12*bg_penalty

                # гістерезис у B
                if mode == 'B':
                    S_ok = S >= max(0.50, 0.90*last_good_score)
                else:
                    S_ok = True

                if S_ok and S > best_S:
                    best_S = S
                    final_bbox = cand
                    meas_ok = True
                    best_res_ratio = res_ratio
                    best_ed_local  = ed_local

            # якщо вимірів не знайшли — адаптивний шум процесу
            if not meas_ok:
                if predict_streak == 0:
                    predict_streak = PREDICT_AHEAD_FRAMES
                predict_streak = max(0, predict_streak - 1)
                scale = min(1.0 + 0.25*(PREDICT_AHEAD_FRAMES - predict_streak), 3.0)
                kf.processNoiseCov = (Q_base * scale).astype(np.float32)

        # -------------------- Корекція Калмана / оновлення шаблону --------------
        if meas_ok:
            mcx,mcy,mw,mh = bbox_to_cxcywh(final_bbox)
            meas = np.array([[mcx],[mcy],[mw],[mh]], np.float32)
            kf.correct(meas)
            predict_streak = 0
            kf.processNoiseCov = Q_base.copy()
            conf = float(np.clip(best_S, 0.0, 1.0))

            # оновлення шаблону — тільки на справді «інформативних» кадрах
            if mw>=T_MIN_SIZE and mh>=T_MIN_SIZE:
                fx,fy,fw,fh = final_bbox
                L_roi_upd = L_full[fy:fy+fh, fx:fx+fw]
                # edge density у блоці оновлення (для підстраховки)
                edges_upd = cv2.Canny(frame[fy:fy+fh, fx:fx+fw], EDGE_LOW, EDGE_HIGH) if fw>2 and fh>2 else None
                ed_upd = float(np.mean(edges_upd>0)) if edges_upd is not None else 0.0
                # умови з тексту: хороша частка залишку, не надто яскраво, є контур
                if (conf >= T_UPDATE_CONF and
                    'best_res_ratio' in locals() and best_res_ratio >= 0.30 and
                    float(np.mean(L_roi_upd)) <= (LAB_L_MAX_GATE - 6.0) and
                    ed_upd >= EDGE_DENS_MIN_B):
                    mx = fx + fw//10; my = fy + fh//10
                    mw2 = max(12, int(fw*0.8)); mh2 = max(12, int(fh*0.8))
                    mx = max(0, min(mx, W-mw2)); my = max(0, min(my, H-mh2))
                    template_ncc = frame[my:my+mh2, mx:mx+mw2].copy()

            # оновити гістерезис
            if mode=='A':
                last_good_score = 0.85*last_good_score + 0.15*conf
        else:
            conf = 0.0

        # -------------------- Visualization panels ------------------------------
        # TL: original with boxes + info
        tl = frame.copy()
        draw_bbox(tl, final_bbox, (0,255,0) if meas_ok else (0,215,255),
                  "MEAS" if meas_ok else "Predict-only")
        draw_bbox(tl, pred_bbox_aff, (200,200,200), "Kalman pred (GMC)")
        put_text(
            tl,
            f"Mode:{mode}  Q:{float(Q):.2f}  Inl:{int(n_inliers)}  E:{E_res:.1f}  p_hi:{p_hi_res:.2f}  edge:{edge_density_W:.2f}",
            (10, 26), 0.6, (255,255,255)
        )
        put_text(tl, f"CONF:{conf:.2f}  FPS:{fps:.1f}", (10, 50), 0.6, (0,255,255))

        # TR: residual flow heatmap
        tr = mag_vis.copy()
        put_text(tr, "Residual flow (TR)", (12, 28), 0.8, (255,255,255))

        # BL: motion mask in W (робустно до розмірів)
        bl = np.zeros_like(frame)
        if motion_mask is not None:
            mm = motion_mask
            if mm.ndim != 2:
                mm = np.zeros((hh, ww), dtype=np.uint8)
            if mm.shape[0] != hh or mm.shape[1] != ww:
                mm = cv2.resize(mm, (ww, hh), interpolation=cv2.INTER_NEAREST)
            mm_rgb = cv2.cvtColor((mm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            bl[y0:y0+hh, x0:x0+ww] = mm_rgb
        draw_bbox(bl, (x0,y0,ww,hh), (100,180,255), "W")
        put_text(bl, "Motion mask (BL)", (12, 28), 0.8, (255,255,255))

        # BR: residual |curr - warp(prev)| (clipped)
        br = br_panel.copy()
        put_text(br, "Residual |curr - warp(prev)| (BR)", (12, 28), 0.7, (255,255,255))

        # 4-панель і запис
        canvas = stack2x2(tl, tr, bl, br)
        cv2.imshow("Tracking (4v)", canvas)
        writer.write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        prev_color = frame.copy()
        prev_gray  = gray.copy()

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved 4-view MP4 to: {out_path}")

if __name__ == "__main__":
    main()
