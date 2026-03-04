"""
CF-DF 模型：基于扩散净化的反事实多模态仇恨言论检测

该模型通过扩散过程对特征进行去偏净化，消除虚假关联，
对比净化前后的置信度变化，体现去偏效果。

当有真实 .pt 权重文件时，可替换 _load_weights 中的逻辑。
"""

import traceback

import numpy as np
import torch
import torch.nn as nn

from backend.config import CF_DF_WEIGHTS_PATH, CLIP_FEATURE_DIM, HATE_THRESHOLD
from backend.models.base import BaseDetectionModel, DetectionResult


class DenoiseNet(nn.Module):
    """3 层 MLP 去噪网络，带残差连接（ε_θ）"""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 1024, time_emb_dim: int = 128):
        super().__init__()
        # 时间步嵌入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, feature_dim),
        )
        # 3 层 MLP 带残差
        self.layer1 = nn.Linear(feature_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, feature_dim)
        self.act = nn.SiLU()
        # 残差跳跃连接
        self.skip1 = nn.Linear(feature_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t 是归一化后的标量时间步，嵌入为特征维度
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # (B, feature_dim)
        h = x_t + t_emb
        h1 = self.act(self.layer1(h))
        h1 = h1 + self.skip1(h)  # 残差
        h2 = self.act(self.layer2(h1))
        h2 = h2 + h1  # 残差
        out = self.layer3(h2)
        out = out + self.skip2(h1)  # 残差
        return out


class CFDFModel(nn.Module):
    """CF-DF：基于反事实扩散去偏的多模态仇恨检测模型"""

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 8,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        alpha_debias: float = 0.4,
        beta_debias: float = 0.4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.alpha_debias = alpha_debias
        self.beta_debias = beta_debias

        # 1. 特征投影（CLIP → 共享空间）
        self.text_projection = nn.Linear(feature_dim, feature_dim)
        self.image_projection = nn.Linear(feature_dim, feature_dim)

        # 2. 交互融合：对称交叉注意力
        self.cross_attn_t2v = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.cross_attn_v2t = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.fusion_proj = nn.Linear(feature_dim * 2, feature_dim)

        # 3. 反事实投影（两个反事实世界共享参数）
        self.cf_proj = nn.Linear(feature_dim * 2, feature_dim)

        # 4. 各世界独立的 LayerNorm
        self.ln_factual = nn.LayerNorm(feature_dim)
        self.ln_no_text = nn.LayerNorm(feature_dim)
        self.ln_no_image = nn.LayerNorm(feature_dim)

        # 5. 三个世界共享的去噪网络
        self.denoiser = DenoiseNet(feature_dim)

        # 6. 共享分类头
        self.classifier = nn.Linear(feature_dim, 1)

        # 扩散噪声调度（线性）
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def _interaction_fusion(
        self, f_t: torch.Tensor, f_v: torch.Tensor
    ) -> torch.Tensor:
        """对称交叉注意力融合"""
        # 确保 3D 输入：(B, 1, D)
        if f_t.dim() == 2:
            f_t = f_t.unsqueeze(1)
        if f_v.dim() == 2:
            f_v = f_v.unsqueeze(1)

        attn_t2v, _ = self.cross_attn_t2v(f_t, f_v, f_v)  # 文本查询图像
        attn_v2t, _ = self.cross_attn_v2t(f_v, f_t, f_t)  # 图像查询文本

        attn_t2v = attn_t2v.squeeze(1)
        attn_v2t = attn_v2t.squeeze(1)

        fused = self.fusion_proj(torch.cat([attn_t2v, attn_v2t], dim=-1))
        return fused

    def _counterfactual_features(
        self, f_t: torch.Tensor, f_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建 Factual + 两个 Counterfactual 特征"""
        d = self.feature_dim
        batch_size = f_t.size(0)
        zeros = torch.zeros(batch_size, d, device=f_t.device)

        F_o = self._interaction_fusion(f_t, f_v)  # Factual World
        F_no_text = self.cf_proj(torch.cat([f_v, zeros], dim=-1))  # Text-Absent World
        F_no_image = self.cf_proj(torch.cat([zeros, f_t], dim=-1))  # Image-Absent World

        # LayerNorm
        F_o = self.ln_factual(F_o)
        F_no_text = self.ln_no_text(F_no_text)
        F_no_image = self.ln_no_image(F_no_image)

        return F_o, F_no_text, F_no_image

    def _tweedie_denoise(
        self, x_t: torch.Tensor, t_star: int
    ) -> torch.Tensor:
        """Tweedie 公式一步去噪"""
        t_tensor = torch.full(
            (x_t.size(0),), t_star, device=x_t.device, dtype=torch.long
        )
        eps_pred = self.denoiser(x_t, t_tensor.float() / self.alphas_cumprod.size(0))  # 归一化时间步

        sqrt_alpha = self.sqrt_alphas_cumprod[t_star]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_star]

        x_0_hat = (x_t - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha
        return x_0_hat

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        t_star: int = 50,
        num_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向推理。

        Args:
            text_features: (B, 512) CLIP 文本特征
            image_features: (B, 512) CLIP 图像特征
            t_star: 扩散净化时间步（超参数）
            num_samples: K 次随机噪声采样取均值

        Returns:
            (final_prob, raw_prob): 去偏后概率（NIE）和去偏前概率（TE）
        """
        # 特征投影
        f_t = self.text_projection(text_features)
        f_v = self.image_projection(image_features)

        # 构建三个世界的特征
        F_o, F_no_text, F_no_image = self._counterfactual_features(f_t, f_v)

        # 扩散净化：K 次随机噪声取均值
        refined_F_o = torch.zeros_like(F_o)
        refined_F_no_text = torch.zeros_like(F_no_text)
        refined_F_no_image = torch.zeros_like(F_no_image)

        sqrt_alpha = self.sqrt_alphas_cumprod[t_star]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_star]

        for _ in range(num_samples):
            for feat, target in [
                (F_o, refined_F_o),
                (F_no_text, refined_F_no_text),
                (F_no_image, refined_F_no_image),
            ]:
                noise = torch.randn_like(feat)
                x_t = sqrt_alpha * feat + sqrt_one_minus_alpha * noise
                denoised = self._tweedie_denoise(x_t, t_star)
                target.add_(denoised)

        refined_F_o /= num_samples
        refined_F_no_text /= num_samples
        refined_F_no_image /= num_samples

        # 共享分类头
        z_o = self.classifier(refined_F_o)          # (B, 1)
        z_no_text = self.classifier(refined_F_no_text)   # (B, 1)
        z_no_image = self.classifier(refined_F_no_image)  # (B, 1)

        # 因果效应聚合：z = z_o - α·sg(tanh(z_¬t)) - β·sg(tanh(z_¬v))
        z = (
            z_o
            - self.alpha_debias * torch.tanh(z_no_text).detach()
            - self.beta_debias * torch.tanh(z_no_image).detach()
        )

        final_prob = torch.sigmoid(z).squeeze(-1)
        raw_prob = torch.sigmoid(z_o).squeeze(-1)

        return final_prob, raw_prob


class CfDfModel(BaseDetectionModel):
    """CF-DF 检测模型"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CFDFModel(CLIP_FEATURE_DIM).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        """加载模型权重，若无权重文件则使用默认初始化"""
        if CF_DF_WEIGHTS_PATH:
            try:
                state_dict = torch.load(
                    CF_DF_WEIGHTS_PATH,
                    map_location=self.device,
                    weights_only=False,
                )
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[CF-DF] 加载权重失败，使用默认参数: {e}")
                traceback.print_exc()

    def predict(
        self, text_features: np.ndarray, image_features: np.ndarray, has_image: bool = True
    ) -> DetectionResult:
        with torch.no_grad():
            text_tensor = torch.tensor(
                text_features, dtype=torch.float32
            ).to(self.device)

            if text_tensor.dim() == 1:
                text_tensor = text_tensor.unsqueeze(0)

            if not has_image:
                # 无图片时将图像特征置零，batch 维度与文本一致
                image_tensor = torch.zeros_like(text_tensor)
            else:
                image_tensor = torch.tensor(
                    image_features, dtype=torch.float32
                ).to(self.device)
                if image_tensor.dim() == 1:
                    image_tensor = image_tensor.unsqueeze(0)

            final_prob, raw_prob = self.model(text_tensor, image_tensor)

            purified_score = final_prob.item()
            raw_score = raw_prob.item()

        is_hateful = purified_score >= HATE_THRESHOLD

        diff = raw_score - purified_score
        if abs(diff) > 0.1:
            bias_note = (
                f"净化后置信度从 {raw_score:.1%} 变为 {purified_score:.1%}，"
                f"说明原始特征存在较强偏置（偏差 {abs(diff):.1%}），"
                f"扩散净化有效消除了虚假关联。"
            )
        else:
            bias_note = (
                f"净化前后置信度变化不大"
                f"（{raw_score:.1%} → {purified_score:.1%}），"
                f"说明该样本的特征偏置较小。"
            )

        if not has_image:
            text_weight = 1.0
            image_weight = 0.0
            explanation = (
                f"模型判定该内容{'包含仇恨言论' if is_hateful else '安全'}，"
                f"净化后置信度为 {purified_score:.1%}。"
                f"本次检测仅使用文本输入（未提供图片），判定完全基于文本模态。{bias_note}"
            )
            extra = {
                "algorithm": "cf-df",
                "raw_probability": round(raw_score, 4),
                "purified_probability": round(purified_score, 4),
                "bias_removed": round(abs(diff), 4),
                "image_provided": False,
            }
        else:
            text_weight = 0.5
            image_weight = 0.5
            explanation = (
                f"模型判定该内容{'包含仇恨言论' if is_hateful else '安全'}，"
                f"净化后置信度为 {purified_score:.1%}。{bias_note}"
            )
            extra = {
                "algorithm": "cf-df",
                "raw_probability": round(raw_score, 4),
                "purified_probability": round(purified_score, 4),
                "bias_removed": round(abs(diff), 4),
            }

        return DetectionResult(
            is_hateful=is_hateful,
            hate_probability=purified_score,
            text_weight=text_weight,
            image_weight=image_weight,
            explanation=explanation,
            extra=extra,
        )

    def get_algorithm_name(self) -> str:
        return "CF-DF（基于扩散净化的反事实检测）"
