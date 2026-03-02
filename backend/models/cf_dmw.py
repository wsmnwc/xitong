"""
CF-DMW 模型：基于动态模态权重的反事实多模态仇恨言论检测（MoRE）

该模型通过 MoRE（Mixture of Routing Experts）动态计算文本模态权重 w_t
和图像模态权重 w_v，实现对不同样本自适应地分配模态贡献度，从而提升检测性能。

当有真实 .pt 权重文件时，可通过环境变量 CF_DMW_WEIGHTS_PATH 指定路径。
"""

import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import CF_DMW_WEIGHTS_PATH, CLIP_FEATURE_DIM, HATE_THRESHOLD
from backend.models.base import BaseDetectionModel, DetectionResult


class ModalExpert(nn.Module):
    """模态专家网络 (text_expert / vision_expert)"""

    def __init__(self, dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),   # ffn.0
            nn.ReLU(),                # ffn.1
            nn.Dropout(p=0.1),       # ffn.2
            nn.Linear(hidden, dim),  # ffn.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class VisionPositionalEncoding(nn.Module):
    """图像位置编码 (vision_pe)"""

    def __init__(self, num_patches: int = 16, dim: int = 512):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(num_patches, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_patches, dim)
        return x + self.positional_encoding


class VisionPooling(nn.Module):
    """图像注意力池化 (vision_pooling)"""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.attention = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_patches, dim)
        attn = torch.softmax(self.attention(x), dim=1)  # (batch, num_patches, 1)
        return (attn * x).sum(dim=1)  # (batch, dim)


class MoREModel(nn.Module):
    """MoRE: Mixture of Routing Experts"""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 1024,
        num_classes: int = 2,
        classifier_hidden: int = 200,
        num_patches: int = 16,
    ):
        super().__init__()
        # 输入投影
        self.text_input_projection = nn.Linear(feature_dim, feature_dim)
        self.vision_input_projection = nn.Linear(feature_dim, feature_dim)
        # 模态专家
        self.text_expert = ModalExpert(feature_dim, hidden_dim)
        self.vision_expert = ModalExpert(feature_dim, hidden_dim)
        # 图像位置编码和池化
        self.vision_pe = VisionPositionalEncoding(num_patches, feature_dim)
        self.vision_pooling = VisionPooling(feature_dim)
        # 路由器 (动态模态权重 w_t, w_v)
        self.router = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # router.0
            nn.ReLU(),                                  # router.1
            nn.Linear(feature_dim, 2),                  # router.2
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden),  # classifier.0
            nn.ReLU(),                                   # classifier.1
            nn.Dropout(p=0.1),                           # classifier.2
            nn.Linear(classifier_hidden, num_classes),   # classifier.3
        )
        # 单模态预测器（用于训练时的反事实损失，推理时不使用）
        self.text_preditor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # text_preditor.0
            nn.ReLU(),                             # text_preditor.1
            nn.Linear(feature_dim, num_classes),   # text_preditor.2
        )
        self.vision_preditor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # vision_preditor.0
            nn.ReLU(),                             # vision_preditor.1
            nn.Linear(feature_dim, num_classes),   # vision_preditor.2
        )

    def forward(
        self, text_features: torch.Tensor, vision_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            text_features:   (batch, dim)
            vision_features: (batch, dim) 或 (batch, num_patches, dim)

        Returns:
            (logits, w_t, w_v): 2-class logits 以及文本/图像路由权重
        """
        text_feat = self.text_input_projection(text_features)
        vision_feat = self.vision_input_projection(vision_features)

        # 若为 patch 级别特征，加位置编码后池化为单向量
        if vision_feat.dim() == 3:
            vision_feat = self.vision_pe(vision_feat)
            vision_feat = self.vision_pooling(vision_feat)

        text_out = self.text_expert(text_feat)
        vision_out = self.vision_expert(vision_feat)

        combined = torch.cat([text_out, vision_out], dim=-1)
        router_logits = self.router(combined)
        router_weights = torch.softmax(router_logits, dim=-1)
        w_t = router_weights[:, 0:1]
        w_v = router_weights[:, 1:2]

        fused = w_t * text_out + w_v * vision_out
        logits = self.classifier(fused)
        return logits, w_t, w_v


class CfDmwModel(BaseDetectionModel):
    """CF-DMW 检测模型"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MoREModel().to(self.device)
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
            except Exception as e:
                print(f"[CF-DMW] 加载权重失败，使用默认参数: {e}")
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

            hate_prob = F.softmax(logits, dim=-1)[:, 1].item()
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
