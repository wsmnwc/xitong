"""
CF-DMW 模型：基于动态模态权重的反事实多模态仇恨言论检测

该模型通过动态计算文本模态权重 w_t 和图像模态权重 w_v，
实现对不同样本自适应地分配模态贡献度，从而提升检测性能。

当有真实 .pth 权重文件时，可替换 _load_weights 中的逻辑。
"""

import numpy as np
import torch
import torch.nn as nn

from backend.config import CF_DMW_WEIGHTS_PATH, CLIP_FEATURE_DIM, HATE_THRESHOLD
from backend.models.base import BaseDetectionModel, DetectionResult


class DynamicWeightFusion(nn.Module):
    """动态权重融合网络：根据输入特征计算各模态权重并融合"""

    def __init__(self, feature_dim: int = CLIP_FEATURE_DIM):
        super().__init__()
        self.text_gate = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.image_gate = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (hate_prob, text_weight, image_weight)
        """
        text_score = self.text_gate(text_features)
        image_score = self.image_gate(image_features)
        weights = torch.softmax(
            torch.cat([text_score, image_score], dim=-1), dim=-1
        )
        w_t = weights[:, 0:1]
        w_v = weights[:, 1:2]
        fused = w_t * text_features + w_v * image_features
        prob = self.classifier(fused)
        return prob, w_t, w_v


class CfDmwModel(BaseDetectionModel):
    """CF-DMW 检测模型"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DynamicWeightFusion(CLIP_FEATURE_DIM).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        """加载模型权重，若无权重文件则使用默认初始化"""
        if CF_DMW_WEIGHTS_PATH:
            try:
                state_dict = torch.load(
                    CF_DMW_WEIGHTS_PATH,
                    map_location=self.device,
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[CF-DMW] 加载权重失败，使用默认参数: {e}")

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

            prob, w_t, w_v = self.model(text_tensor, image_tensor)

            hate_prob = prob.item()
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
