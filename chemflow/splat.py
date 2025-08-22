# FLIP/chemflow/splat.py
import torch

def gaussian_splat(pts, values, H, W, sigma=5.0, device=None):
    """
    把 N 个点的标量 values 高斯扩散到 (H,W) 网格，返回 (1,1,H,W)
    pts: (N,2) 像素坐标 (x,y)
    values: (N,) 标量
    sigma: 高斯半径像素
    """
    if device is None:
        device = values.device if isinstance(values, torch.Tensor) else torch.device("cpu")
    if not isinstance(pts, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=torch.float32, device=device)
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.float32, device=device)

    heat = torch.zeros((1,1,H,W), device=device)
    if pts.numel() == 0:
        return heat

    xs = pts[:,0].clamp(0, W-1)
    ys = pts[:,1].clamp(0, H-1)
    xv = torch.arange(W, device=device).view(1,1,1,W)
    yv = torch.arange(H, device=device).view(1,1,H,1)

    # 逐点累积（简洁实现；如需更快可做局部窗口化）
    for i in range(pts.shape[0]):
        dx2 = (xv - xs[i])**2
        dy2 = (yv - ys[i])**2
        g = torch.exp(-(dx2 + dy2)/(2*sigma**2)) * values[i]
        heat += g
    # 归一化到 0~1
    heat = (heat - heat.amin()) / (heat.amax() - heat.amin() + 1e-8)
    return heat
