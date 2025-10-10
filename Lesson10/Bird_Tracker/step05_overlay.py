import cv2, json
import numpy as np
from utils import progress_iter, write_video

def run(BASE, cfg, src_video_path, track_json_path, transforms_json_path, out_mp4):
    cap = cv2.VideoCapture(str(src_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {src_video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks = json.load(open(track_json_path, "r", encoding="utf-8"))["tracks"]
    bbmap = {t["frame"]:(t["x"],t["y"],t["w"],t["h"]) for t in tracks}

    T = json.load(open(transforms_json_path, "r", encoding="utf-8"))
    C_list = T.get("C_to_ref", [])
    # у TR-режимі C_t = I; загальна формула:
    # маємо bbox у координатах термокадру (які = оригінал), тож просто малюємо.

    frames=[]
    for i in progress_iter(range(N), total=N, desc="Step5 Overlay"):
        ok, fr = cap.read()
        if not ok: break
        if i in bbmap:
            x,y,w,h = bbmap[i]
            cv2.rectangle(fr, (x,y),(x+w,y+h), (0,255,255), 2)
        frames.append(fr)
    cap.release()
    write_video(frames, out_mp4, (W,H), fps, desc="Write overlay_tracked")
