# FLIP/scripts/train_flip_with_chemflow.py
import yaml, torch
from torch.utils.data import DataLoader
from datasets.libero_chemflow import LiberoChemflowDataset
# 假设你已有 action/dynamics/value 的训练脚本或 Trainer，这里只展示如何接入 query_pts

def build_dinov3():
    from models.dinov3 import build_dinov3_backbone
    return build_dinov3_backbone()

def main():
    cfg = yaml.safe_load(open("config/flip_with_chemflow.yaml"))
    dinov3 = build_dinov3()
    ds = LiberoChemflowDataset(dinov3=dinov3, text_encoder=None,
                               img_size=cfg["img_size"], N=cfg["N"])
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    # 伪代码：取出你已有的模型
    action_model = ...
    dynamics_model = ...
    value_model = ...

    optim = torch.optim.AdamW([...], lr=1e-4)

    for epoch in range(20):
        for batch in dl:
            img = batch["img"]                     # (B,3,H,W)
            pts = batch["query_pts"]               # (B,N,2)
            # 把 pts 作为本 step 的查询点，送入 action_model
            # flows_pred = action_model(img, pts, lang=...)
            # loss = ...
            # optim.zero_grad(); loss.backward(); optim.step()
            pass

if __name__ == "__main__":
    main()
