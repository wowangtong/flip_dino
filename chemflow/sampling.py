# FLIP/chemflow/sampling.py
import numpy as np
import torch

def sample_query_points_from_heatmap(
    H, N=529, min_dist=8, keep_uniform_ratio=0.15, temperature=0.7, seed=None
):
    """
    H: (1,1,H,W) torch tensor [0,1]
    返回: (N,2) numpy int32, (x,y) 像素坐标
    """
    if seed is not None:
        np.random.seed(seed)

    H = H[0,0].detach().cpu().float().numpy()
    H = H / (H.sum() + 1e-8)
    H = np.power(H, 1.0/max(1e-6, temperature))
    H = H / (H.sum() + 1e-8)

    Hh, Ww = H.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    weights = H.reshape(-1)

    num_uniform = int(N * keep_uniform_ratio)
    num_weighted = N - num_uniform

    # 均匀采样（粗网格）
    if num_uniform > 0:
        stride = max(1, int(np.sqrt((Hh*Ww)/max(num_uniform,1))))
        grid_x = np.arange(0, Ww, stride)
        grid_y = np.arange(0, Hh, stride)
        gxx, gyy = np.meshgrid(grid_x, grid_y)
        grid_coords = np.stack([gxx.ravel(), gyy.ravel()], axis=-1)
        if len(grid_coords) > num_uniform:
            sel = np.random.choice(len(grid_coords), size=num_uniform, replace=False)
            uni_pts = grid_coords[sel]
        else:
            uni_pts = grid_coords
    else:
        uni_pts = np.zeros((0,2), dtype=np.int32)

    # 按权重采样 + 最小距离
    idx_all = np.arange(coords.shape[0])
    chosen = []
    picked = set()
    cand = np.random.choice(idx_all, size=min(num_weighted*6, coords.shape[0]), replace=False, p=weights)

    for i in cand:
        p = coords[i]
        if len(chosen) == 0:
            chosen.append(p); picked.add(i); continue
        d2 = ((np.array(chosen) - p)**2).sum(axis=1)
        if (d2.min() if len(d2)>0 else 1e9) >= (min_dist**2):
            chosen.append(p); picked.add(i)
        if len(chosen) >= num_weighted:
            break

    if len(chosen) < num_weighted:
        remain = [i for i in idx_all if i not in picked]
        extra = np.random.choice(remain, size=(num_weighted - len(chosen)), replace=False)
        chosen += [coords[i] for i in extra]

    pts = np.vstack([uni_pts, np.array(chosen, dtype=np.int32)])
    if len(pts) > N:
        sel = np.random.choice(len(pts), size=N, replace=False)
        pts = pts[sel]
    return pts.astype(np.int32)  # (x,y)
