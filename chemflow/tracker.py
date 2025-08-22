# FLIP/chemflow/tracker.py
import torch
import numpy as np

class CoTrackerWrapper:
    """
    轻封装：给定连续帧与起点，输出未来 L 步的点轨迹。
    要求已有 cotracker 安装好；若你在 scripts/video_tracking.py 里已经有加载代码，
    这里可以直接 import 你那边的 predictor / model 构造函数来用。
    """
    def __init__(self, predictor_ctor=None, device="cuda"):
        """
        predictor_ctor: 一个可调用，返回 predictor 对象，具备
          predict(frames: (T,3,H,W) [0..1], points_xy: (N,2)) -> traj: (T,N,2)
        """
        self.device = device
        if predictor_ctor is None:
            # 占位：你可替换为项目里现成的构造
            self.predictor = None
        else:
            self.predictor = predictor_ctor().to(device)

    @torch.no_grad()
    def track_points(self, frames, points_xy):
        """
        frames: (T,3,H,W) torch float [0..1]
        points_xy: (N,2) numpy int/float (x,y) 在 frames[0] 上的坐标
        返回: traj: (T,N,2) numpy float
        """
        if self.predictor is None:
            # 这里给个简单占位：恒等轨迹（只为跑通可视化），请替换为真正的 CoTracker 推理
            T, _, _, _ = frames.shape
            pts = np.asarray(points_xy, dtype=np.float32)
            traj = np.stack([pts for _ in range(T)], axis=0)  # (T,N,2)
            return traj
        else:
            # 调用你已有的 predictor
            traj = self.predictor.predict(frames, points_xy)  # (T,N,2)
            return traj
