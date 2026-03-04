"""基础模型抽象类，定义检测模型的统一接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """检测结果数据类"""

    is_hateful: bool  # 是否为仇恨言论
    hate_probability: float  # 仇恨概率 [0, 1]
    text_weight: float  # 文本模态权重
    image_weight: float  # 图像模态权重
    explanation: str  # 可解释性说明
    extra: dict | None = None  # 额外信息（如去偏前后置信度等）


class BaseDetectionModel(ABC):
    """检测模型基类"""

    @abstractmethod
    def predict(
        self, text_features: np.ndarray, image_features: np.ndarray, has_image: bool = True
    ) -> DetectionResult:
        """
        对输入的文本和图像特征进行仇恨言论检测。

        Args:
            text_features: CLIP 提取的文本特征向量
            image_features: CLIP 提取的图像特征向量
            has_image: 用户是否提供了真实图片（False 时图像特征为占位图生成）

        Returns:
            DetectionResult: 检测结果
        """
        ...

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """返回算法名称"""
        ...
