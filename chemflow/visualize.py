# FLIP/chemflow/visualize.py
import cv2
import numpy as np
import torch

def overlay_heatmap(img_bgr, H, alpha=0.5):
    """
    img_bgr: uint8 (H,W,3)
    H: torch (1,1,H,W) 0..1
    """
    h = (H[0,0].detach().cpu().numpy() * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 1.0, h_color, alpha, 0)
    return out

def draw_points(img_bgr, pts, color=(0,255,0)):
    vis = img_bgr.copy()
    for (x,y) in pts:
        cv2.circle(vis, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    return vis
