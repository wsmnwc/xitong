"""
CF-DMW 模型：基于动态模态权重的反事实多模态仇恨言论检测 (MoRE)

该模型通过路由器动态计算文本模态权重 w_t 和图像模态权重 w_v，
实现对不同样本自适应地分配模态贡献度，从而提升检测性能。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import CF_DMW_WEIGHTS_PATH, CLIP_FEATURE_DIM, HATE_THRESHOLD
from backend.models.base import BaseDetectionModel, DetectionResult


class ModalExpert(nn.Module):
    """模态专家网络"""

    def __init__(self, dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)  # residual connection


class VisionPositionalEncoding(nn.Module):
    """图像位置编码"""

    def __init__(self, num_patches: int = 16, dim: int = 512):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(num_patches, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_patches, dim) or (batch, dim)
        if x.dim() == 2:
            return x  # single vector, skip PE
        return x + self.positional_encoding[: x.size(1)]


class VisionPooling(nn.Module):
    """图像注意力池化"""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_patches, dim) or (batch, dim)
        if x.dim() == 2:
            return x  # already a single vector
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, num_patches, 1)
        pooled = (attn_weights * x).sum(dim=1)  # (batch, dim)
        return pooled


class MoREModel(nn.Module):
    """
    MoRE: Mixture of Routing Experts

    Architecture matching MoRE_MAMI_best.pt state_dict (31 parameters).
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
        num_classes: int = 2,
        classifier_hidden: int = 200,
        num_patches: int = 16,
    ):
        super().__init__()
        # Input projections
        self.text_input_projection = nn.Linear(feature_dim, feature_dim)
        self.vision_input_projection = nn.Linear(feature_dim, feature_dim)

        # Modal experts
        self.text_expert = ModalExpert(feature_dim, hidden_dim)
        self.vision_expert = ModalExpert(feature_dim, hidden_dim)

        # Vision positional encoding and pooling
        self.vision_pe = VisionPositionalEncoding(num_patches, feature_dim)
        self.vision_pooling = VisionPooling(feature_dim)

        # Router: produces dynamic modal weights [w_t, w_v]
        self.router = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes),
        )

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden, num_classes),
        )

        # Unimodal predictors (for counterfactual training, needed for state_dict)
        self.text_preditor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes),
        )
        self.vision_preditor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            text_features: (batch, feature_dim)
            image_features: (batch, feature_dim)

        Returns:
            (hate_logits, w_t, w_v)
        """
        # Project inputs
        text_proj = self.text_input_projection(text_features)
        vision_proj = self.vision_input_projection(image_features)

        # Expert processing with residual
        text_out = self.text_expert(text_proj)
        vision_out = self.vision_expert(vision_proj)

        # Vision PE + pooling (handles single-vector input gracefully)
        vision_out = self.vision_pe(vision_out)
        vision_out = self.vision_pooling(vision_out)

        # Router: compute dynamic modal weights
        combined = torch.cat([text_out, vision_out], dim=-1)
        router_logits = self.router(combined)  # (batch, 2)
        router_weights = F.softmax(router_logits, dim=-1)
        w_t = router_weights[:, 0:1]  # (batch, 1)
        w_v = router_weights[:, 1:2]  # (batch, 1)

        # Weighted fusion
        fused = w_t * text_out + w_v * vision_out

        # Classification
        logits = self.classifier(fused)  # (batch, 2)

        return logits, w_t, w_v


class CfDmwModel(BaseDetectionModel):
    """CF-DMW 检测模型（MoRE 架构）"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MoREModel(CLIP_FEATURE_DIM).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        """加载模型权重，若无权重文件则使用默认初始化"""
        if CF_DMW_WEIGHTS_PATH:
            try:
                state_dict = torch.load(
                    CF_DMW_WEIGHTS_PATH,
                    map_location=self.device,
                    weights_only=False,
                )
                self.model.load_state_dict(state_dict)
                print("[CF-DMW] ✅ 模型权重加载成功!")
            except Exception as e:
                print(f"[CF-DMW] ❌ 加载权重失败，使用默认参数: {e}")
                import traceback
                traceback.print_exc()

    def predict(
        self, text_features: np.ndarray, image_features: np.ndarray
    ) -> DetectionResult:
        with torch.no_grad():
            text_tensor = torch.tensor(
                text_features, dtype=torch.float32
            ).to(self.device)
            image_tensor = torch.tensor(
                image_features, dtype=torch.float32
            ).to(self.device)

            if text_tensor.dim() == 1:
                text_tensor = text_tensor.unsqueeze(0)
            if image_tensor.dim() == 1:
                image_tensor = image_tensor.unsqueeze(0)

            logits, w_t, w_v = self.model(text_tensor, image_tensor)

            # 2-class output: softmax then take class-1 (hateful) probability
            probs = F.softmax(logits, dim=-1)
            hate_prob = probs[:, 1].item()
            text_weight = w_t.item()
            image_weight = w_v.item()

        is_hateful = hate_prob >= HATE_THRESHOLD

        if text_weight > image_weight:
            dominant = "文本"
            ratio = text_weight / (text_weight + image_weight) * 100
        else:
            dominant = "图像"
            ratio = image_weight / (text_weight + image_weight) * 100

        explanation = (
            f"模型判定该内容{'包含仇恨言论' if is_hateful else '安全'}，"
            f"置信度为 {hate_prob:.1%}。"
            f"在本次判定中，{dominant}模态贡献占比 {ratio:.1f}%，"
            f"是影响判定结果的主要因素。"
        )

        return DetectionResult(
            is_hateful=is_hateful,
            hate_probability=hate_prob,
            text_weight=text_weight,
            image_weight=image_weight,
            explanation=explanation,
            extra={
                "algorithm": "cf-dmw",
                "dynamic_weights": {
                    "w_t": round(text_weight, 4),
                    "w_v": round(image_weight, 4),
                },
            },
        )

    def get_algorithm_name(self) -> str:
        return "CF-DMW（基于动态模态权重的反事实检测）"
