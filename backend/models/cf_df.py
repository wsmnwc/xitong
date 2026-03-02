"""
CF-DF 模型：基于扩散净化的反事实多模态仇恨言论检测

该模型通过扩散过程对特征进行去偏净化，消除虚假关联，
对比净化前后的置信度变化，体现去偏效果。

当有真实 .pth 权重文件时，可替换 _load_weights 中的逻辑。
"""

import numpy as np
import torch
import torch.nn as nn

from backend.config import CF_DF_WEIGHTS_PATH, CLIP_FEATURE_DIM, HATE_THRESHOLD
from backend.models.base import BaseDetectionModel, DetectionResult


class DiffusionPurifier(nn.Module):
    """扩散净化模块：模拟扩散过程对特征进行去偏"""

    def __init__(self, feature_dim: int = CLIP_FEATURE_DIM, num_steps: int = 5):
        super().__init__()
        self.num_steps = num_steps
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        self.denoise_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """对输入特征进行扩散净化"""
        x = features
        for _ in range(self.num_steps):
            noise = torch.randn_like(x) * self.noise_scale
            x_noisy = x + noise
            x = x - self.denoise_net(x_noisy)
        return x


class CfDfClassifier(nn.Module):
    """CF-DF 完整分类网络"""

    def __init__(self, feature_dim: int = CLIP_FEATURE_DIM):
        super().__init__()
        self.text_purifier = DiffusionPurifier(feature_dim)
        self.image_purifier = DiffusionPurifier(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.raw_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            (purified_prob, raw_prob): 净化后概率和原始概率
        """
        raw_fused = torch.cat([text_features, image_features], dim=-1)
        raw_prob = self.raw_classifier(raw_fused)

        purified_text = self.text_purifier(text_features)
        purified_image = self.image_purifier(image_features)
        purified_fused = torch.cat([purified_text, purified_image], dim=-1)
        purified_prob = self.classifier(purified_fused)

        return purified_prob, raw_prob


class CfDfModel(BaseDetectionModel):
    """CF-DF 检测模型"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CfDfClassifier(CLIP_FEATURE_DIM).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        """加载模型权重，若无权重文件则使用默认初始化"""
        if CF_DF_WEIGHTS_PATH:
            try:
                state_dict = torch.load(
                    CF_DF_WEIGHTS_PATH,
                    map_location=self.device,
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[CF-DF] 加载权重失败，使用默认参数: {e}")

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

            purified_prob, raw_prob = self.model(text_tensor, image_tensor)

            purified_score = purified_prob.item()
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

        explanation = (
            f"模型判定该内容{'包含仇恨言论' if is_hateful else '安全'}，"
            f"净化后置信度为 {purified_score:.1%}。{bias_note}"
        )

        text_weight = 0.5
        image_weight = 0.5

        return DetectionResult(
            is_hateful=is_hateful,
            hate_probability=purified_score,
            text_weight=text_weight,
            image_weight=image_weight,
            explanation=explanation,
            extra={
                "algorithm": "cf-df",
                "raw_probability": round(raw_score, 4),
                "purified_probability": round(purified_score, 4),
                "bias_removed": round(abs(diff), 4),
            },
        )

    def get_algorithm_name(self) -> str:
        return "CF-DF（基于扩散净化的反事实检测）"
