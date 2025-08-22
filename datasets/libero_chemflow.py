# FLIP/datasets/libero_chemflow.py
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

from chemflow.selector import ChemFlowSelector
from chemflow.sampling import sample_query_points_from_heatmap

class LiberoChemflowDataset(Dataset):
    """
    假设 data/libero_10 下有 10 个 hdf5 (你已完成步骤1/2)，并且能取到 agentview 图像与文本。
    这里只示范最小读取：每个样本给出一帧 + 指令文本（如无文本可置空）。
    """
    def __init__(self, root="data/libero_10", dinov3=None, text_encoder=None,
                 img_key="observations/agentview_image", instr_key="language",
                 img_size=256, N=529):
        super().__init__()
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".hdf5")]
        assert len(self.files)>0, f"No hdf5 files found in {root}"
        self.dinov3 = dinov3
        self.text_encoder = text_encoder
        self.selector = ChemFlowSelector(dinov3, img_size=img_size)
        self.img_key = img_key
        self.instr_key = instr_key
        self.N = N
        self.img_size = img_size

        # 索引所有 (file, ep, t) 指针（示例：取每个 episode 的中间一帧）
        self.index = []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                E = len(f["data"])
                for e in range(E):
                    T = len(f[f"data/{e}/{self.img_key}"])
                    t = T//2
                    self.index.append((fp, e, t))

    def __len__(self):
        return len(self.index)

    def _to_tensor_img(self, img):
        # 假设存储为 (H,W,3) uint8
        import cv2
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2,0,1).float()/255.0
        return img

    def __getitem__(self, i):
        fp, e, t = self.index[i]
        with h5py.File(fp, "r") as f:
            img = f[f"data/{e}/{self.img_key}"][t]  # (H,W,3) uint8
            try:
                instr = f[f"data/{e}/{self.instr_key}"][()].decode("utf-8")
            except Exception:
                instr = ""

        img_t = self._to_tensor_img(img)  # (3,H,W)
        if self.text_encoder is not None and len(instr)>0:
            lang = self.text_encoder(instr)  # (T_l, D_l) or (1,T_l,D_l)
            if lang.dim()==2: lang = lang.unsqueeze(0)
        else:
            lang = None

        with torch.no_grad():
            H = self.selector(img_t.unsqueeze(0), lang)  # (1,1,H,W)
        pts = sample_query_points_from_heatmap(H, N=self.N, min_dist=8)

        return {
            "img": img_t,                    # (3,H,W)
            "heatmap": H.squeeze(0),         # (1,H,W)
            "query_pts": torch.from_numpy(pts).float(),  # (N,2) (x,y)
            "instr": instr
        }
