# FLIP/scripts/train_chemflow_head.py
import os, yaml, torch
from torch import nn
from torch.utils.data import DataLoader
from datasets.libero_chemflow import LiberoChemflowDataset
from chemflow.selector import ChemFlowSelector
from chemflow.splat import gaussian_splat
from chemflow.tracker import CoTrackerWrapper
from chemflow.losses import motion_supervision_loss, sparsity_loss, diversity_loss

def dummy_text_encoder(s): return None

def build_dinov3():
    from models.dinov3 import build_dinov3_backbone
    return build_dinov3_backbone()

def build_tracker():
    # 如果你已有 predictor 构造器，可在这里返回
    return CoTrackerWrapper(predictor_ctor=None)  # 先用占位（恒等轨迹），跑通流程；之后替换

def train_one_epoch(dl, selector, tracker, opt, device, cfg):
    selector.train()
    tot = {"motion":0.0, "sparse":0.0, "div":0.0}
    for batch in dl:
        img = batch["img"].to(device)                # (B,3,H,W)
        B, _, H, W = img.shape
        instrs = [x for x in batch["instr"]] if "instr" in batch else [""]*B
        lang = None  # 如需语言，替换为真实 text encoder

        # 前向生成 H
        H_map = selector(img, lang)                  # (B,1,H,W)

        # ---- 生成 motion_map （用 tracker 把每个样本的采样点向后追踪）----
        # 训练头阶段，简单起见：取每个样本的 query_pts 作为起点，并造一个 3 帧的 dummy clip
        # 你可改为从 hdf5 读取 t..t+L 帧，这里为最小闭环示例
        pts0 = batch["query_pts"].numpy()            # (B,N,2)——来自 Dataset 中基于 H 的采点；首次 epoch 可随机或复用 batch 内
        motion_maps = []
        for b in range(B):
            # 伪造一个 3 帧 clip（恒等），实际应读取相邻帧并调用 tracker.track_points
            frames = img[b:b+1].repeat(3,1,1,1)      # (T=3,3,H,W) 0..1
            traj = tracker.track_points(frames, pts0[b])  # (T,N,2)
            # 用轨迹位移能量 splat 到像素网格
            disp = ((traj - traj[0:1])**2).sum(-1)**0.5   # (T,N)
            energy = torch.tensor(disp.sum(0), dtype=torch.float32, device=device)  # (N,)
            mm = gaussian_splat(pts0[b], energy, H, W, sigma=cfg["loss"]["sigma"], device=device)  # (1,1,H,W)
            motion_maps.append(mm)
        motion_map = torch.cat(motion_maps, dim=0)   # (B,1,H,W)

        # 损失
        Lm = motion_supervision_loss(H_map, motion_map)
        Ls = sparsity_loss(H_map, alpha=cfg["loss"]["sparsity_alpha"])
        Ld = diversity_loss(H_map, k=cfg["loss"]["diversity_k"])
        loss = Lm + Ls + Ld

        opt.zero_grad()
        loss.backward()
        opt.step()

        tot["motion"] += float(Lm.detach().cpu())
        tot["sparse"] += float(Ls.detach().cpu())
        tot["div"] += float(Ld.detach().cpu())

    n = max(1,len(dl))
    return {k:v/n for k,v in tot.items()}

def main():
    cfg = yaml.safe_load(open("config/chemflow/train_head.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dinov3 = build_dinov3()
    selector = ChemFlowSelector(dinov3, img_size=cfg["dataset"]["img_size"]).to(device)
    ds = LiberoChemflowDataset(
        root=cfg["dataset"]["root"], dinov3=dinov3, text_encoder=None,
        img_size=cfg["dataset"]["img_size"], N=cfg["dataset"]["N"]
    )
    dl = DataLoader(ds, batch_size=cfg["optim"]["batch_size"], shuffle=True, num_workers=4, drop_last=True)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, selector.parameters()),
        lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"]
    )
    tracker = build_tracker()

    os.makedirs("outputs/chemflow_head", exist_ok=True)
    for ep in range(cfg["optim"]["epochs"]):
        stats = train_one_epoch(dl, selector, tracker, opt, device, cfg)
        print(f"[ep {ep}] motion={stats['motion']:.4f} sparse={stats['sparse']:.4f} div={stats['div']:.4f}")
        torch.save(selector.state_dict(), f"outputs/chemflow_head/selector_ep{ep}.pth")

if __name__ == "__main__":
    main()
