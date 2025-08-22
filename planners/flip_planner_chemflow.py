# FLIP/planners/flip_planner_chemflow.py
import torch
from chemflow.selector import ChemFlowSelector
from chemflow.sampling import sample_query_points_from_heatmap

class FlipPlannerWithChemFlow:
    """
    示例：在 plan_step 前，用 ChemFlow 取 query points，然后调用原本 π_f / D / V。
    这里用占位接口展示如何取 pts 并传给 action 模块。
    """
    def __init__(self, action_model, dyn_model, val_model, dinov3, text_encoder, N=529):
        self.action = action_model
        self.dyn = dyn_model
        self.val = val_model
        self.selector = ChemFlowSelector(dinov3)
        self.text_encoder = text_encoder
        self.N = N

    @torch.no_grad()
    def plan_step(self, obs_hist_imgs, instr):
        # obs_hist_imgs: list[Tensor (3,H,W)]
        img_t = obs_hist_imgs[-1].unsqueeze(0)  # (1,3,H,W)
        lang = self.text_encoder(instr) if self.text_encoder is not None else None
        if lang is not None and lang.dim()==2: lang = lang.unsqueeze(0)

        H = self.selector(img_t, lang)          # (1,1,H,W)
        pts = sample_query_points_from_heatmap(H, N=self.N, min_dist=8)

        # 下游调用（示例；替换为你仓库已有接口）
        flows = self.action(obs_hist_imgs, pts, lang)  # (L,N,2)
        video = self.dyn(obs_hist_imgs, flows, lang)   # 合成视频
        score = self.val(video, instr)                 # value 打分
        return flows, video, score, H, pts
