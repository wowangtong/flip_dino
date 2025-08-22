# FLIP/chemflow/selector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemFlowSelector(nn.Module):
    """
    输入:
      img: (B,3,H,W) 当前帧（可只用最新帧）
      lang_tokens: (B,T_l,D_l) 文本编码（可以 None）
    输出:
      H: (B,1,H,W) 归一化的热力图 (0~1)
    说明:
      - 冻结 DINOv3 主干，只训练投影/交叉注意力/打分头。
    """
    def __init__(
        self,
        dinov3_backbone,
        img_size=256,
        patch_size=16,
        d_lang=4096,
        d_model=768,
        heads=4,
        use_language=True
    ):
        super().__init__()
        self.backbone = dinov3_backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.patch_size = patch_size
        self.img_size = img_size
        self.d_model = d_model
        self.use_language = use_language

        # 假设 DINOv3 提供 embed_dim
        embed_dim = getattr(self.backbone, "embed_dim", d_model)
        self.proj_img = nn.Linear(embed_dim, d_model, bias=False)

        self.proj_lang = nn.Sequential(
            nn.Linear(d_lang, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)

        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, 1)
        )
        self.register_parameter("global_q", nn.Parameter(torch.randn(1, 1, d_model)))

    @torch.no_grad()
    def _extract_patches(self, img):
        """
        返回 (B, P, C)。不同 DINOv3 实现请按需改这段。
        要求 backbone.forward_features(img) -> dict/tuple，含 patch tokens。
        """
        feats = self.backbone.forward_features(img)
        # 兼容不同风格：如果是 (B, P+1, C) 带 CLS
        if isinstance(feats, dict) and "x" in feats:
            x = feats["x"]
        else:
            x = feats  # (B, P+1, C) or (B,P,C)
        if x.shape[1] == 1:
            raise ValueError("DINOv3 backbone returned only CLS token. Need patch tokens.")
        if x.shape[1] == ((img.shape[-1] // self.patch_size) * (img.shape[-2] // self.patch_size) + 1):
            x = x[:, 1:, :]  # 去 CLS
        return x  # (B, P, C)

    def forward(self, img, lang_tokens=None):
        B, _, H, W = img.shape
        P_h, P_w = H // self.patch_size, W // self.patch_size
        tokens = self._extract_patches(img)                 # (B, P, C_img)
        tokens = self.proj_img(tokens)                      # (B, P, d_model)

        if (not self.use_language) or (lang_tokens is None):
            q = self.global_q.expand(B, -1, -1)             # (B,1,d_model)
        else:
            if lang_tokens.dim() == 2:
                lang_tokens = lang_tokens.unsqueeze(0)
            lang_mean = lang_tokens.mean(dim=1)             # (B, d_lang)
            q = self.proj_lang(lang_mean).unsqueeze(1)      # (B,1,d_model)

        attn_out, _ = self.cross_attn(q, tokens, tokens)    # (B,1,d_model)
        gated = tokens * attn_out                           # (B,P,d_model)

        scores = self.score_head(gated)                     # (B,P,1)
        scores = scores.view(B, P_h, P_w, 1).permute(0,3,1,2)  # (B,1,P_h,P_w)
        H_map = F.interpolate(scores, size=(H, W), mode="bilinear", align_corners=False)
        H_map = torch.sigmoid(H_map)

        eps = 1e-6
        H_min = H_map.amin(dim=(2,3), keepdim=True)
        H_max = H_map.amax(dim=(2,3), keepdim=True)
        H_norm = (H_map - H_min) / (H_max - H_min + eps)
        return H_norm
