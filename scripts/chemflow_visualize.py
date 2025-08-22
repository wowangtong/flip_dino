# FLIP/scripts/chemflow_visualize.py
import os, cv2, torch
from datasets.libero_chemflow import LiberoChemflowDataset
from chemflow.visualize import overlay_heatmap, draw_points

def dummy_text_encoder(s):  # 如已有 LLM 编码器可替换
    return None

def build_dinov3():
    # 你在仓库里已有的构造函数；这里给个占位
    from models.dinov3 import build_dinov3_backbone
    return build_dinov3_backbone()

def main():
    os.makedirs("outputs/chemflow_vis", exist_ok=True)
    dinov3 = build_dinov3()
    ds = LiberoChemflowDataset(dinov3=dinov3, text_encoder=dummy_text_encoder, img_size=256, N=529)
    item = ds[0]
    img = (item["img"].permute(1,2,0).numpy()*255).astype("uint8")[:, :, ::-1]  # BGR
    H = item["heatmap"].unsqueeze(0)  # (1,1,H,W)
    pts = item["query_pts"].numpy().astype(int)

    vis_h = overlay_heatmap(img, H)
    vis_pts = draw_points(vis_h, pts)
    cv2.imwrite("outputs/chemflow_vis/vis_0.png", vis_pts)
    print("Saved to outputs/chemflow_vis/vis_0.png")

if __name__ == "__main__":
    main()
