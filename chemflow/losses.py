# FLIP/chemflow/losses.py
import torch
import torch.nn.functional as F

def normalize01(x, eps=1e-6):
    return (x - x.amin(dim=(2,3), keepdim=True)) / (x.amax(dim=(2,3), keepdim=True) - x.amin(dim=(2,3), keepdim=True) + eps)

def motion_supervision_loss(H, motion_map):
    """
    H: (B,1,H,W)
    motion_map: (B,1,H,W) 由轨迹能量 splat 得到
    """
    Hn = normalize01(H)
    Mn = normalize01(motion_map)
    return F.mse_loss(Hn, Mn)

def sparsity_loss(H, alpha=0.05):
    return alpha * H.mean()

def diversity_loss(H, k=256):
    B, _, Hh, Ww = H.shape
    Hf = H.view(B, -1)
    vals, idx = torch.topk(Hf, k=min(k, Hf.shape[1]), dim=1)
    xs = (idx % Ww).float(); ys = (idx // Ww).float()
    d2 = (xs.unsqueeze(2)-xs.unsqueeze(1))**2 + (ys.unsqueeze(2)-ys.unsqueeze(1))**2
    loss = 1.0 / (d2 + 1.0).mean()
    return loss
