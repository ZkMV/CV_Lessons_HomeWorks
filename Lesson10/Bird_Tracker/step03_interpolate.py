import cv2
import json
import numpy as np
# Додаємо DATA_DIR до імпорту
from utils import progress_iter, write_video, DATA_DIR

def run(BASE, cfg, thermo_mp4_path, classif_json_path, out_restored_mp4):
    lab = json.load(open(classif_json_path, "r", encoding="utf-8"))["labels"]

    cap = cv2.VideoCapture(str(thermo_mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {thermo_mp4_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # читаємо усі кадри в RAM
    frames = []
    for _ in progress_iter(range(N), total=N, desc="Load thermo"):
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    N = len(frames)
    if len(lab) != N: lab = (lab + ["good"]*N)[:N]

    # індекси good
    good_idx = [i for i,s in enumerate(lab) if s=="good"]
    if not good_idx:
        out = frames
    else:
        out = frames.copy()

        # заповнення прогалин між good кадрами
        for a, b in zip(good_idx, good_idx[1:]):
            K = b - a
            if K <= 1: continue
            left = frames[a].astype(np.float32)
            right= frames[b].astype(np.float32)
            for j in range(1, K):
                t = j / K
                alpha = (1 - t)**2
                beta  = t**2
                # Використовуємо зважене додавання, щоб уникнути переповнення
                synth = cv2.addWeighted(left, alpha, right, beta, 0.0)
                if lab[a+j] == "bad":
                    out[a+j] = synth.astype(np.uint8)

        # лівий/правий «хвости»
        first = good_idx[0]
        for i in range(0, first):
            if lab[i] == "bad": out[i] = frames[first]
        last = good_idx[-1]
        for i in range(last+1, N):
            if lab[i] == "bad": out[i] = frames[last]

    # +++ НОВИЙ КОД: Створюємо папку та зберігаємо кадри етапу 3 +++
    interpolate_frames_dir = DATA_DIR / "interpolate_frames"
    interpolate_frames_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in progress_iter(enumerate(out), total=len(out), desc="Saving interpolated frames"):
        filename = f"{i:03d}.jpg"
        filepath = interpolate_frames_dir / filename
        cv2.imwrite(str(filepath), frame)
    # +++ КІНЕЦЬ НОВОГО КОДУ +++

    write_video(out, out_restored_mp4, (W,H), fps, desc="Write 3.thermo_restored")