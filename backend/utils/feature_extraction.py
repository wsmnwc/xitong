"""
CLIP 特征提取模块

使用 CLIP 模型提取文本和图像的特征向量。
如果 transformers / CLIP 未安装或无法加载，
则回退到基于哈希的确定性模拟特征，保证系统可运行。
"""

import numpy as np
from PIL import Image

from backend.config import CLIP_FEATURE_DIM, CLIP_MODEL_NAME, DEVICE


class FeatureExtractor:
    """CLIP 特征提取器"""

    def __init__(self) -> None:
        self.device = DEVICE
        self.feature_dim = CLIP_FEATURE_DIM
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._use_real_clip = False
        self._init_clip()

    def _init_clip(self) -> None:
        """尝试加载 CLIP 模型"""
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self._model.to(self.device)
            self._model.eval()
            self._use_real_clip = True
            print(f"[FeatureExtractor] CLIP 模型加载成功: {CLIP_MODEL_NAME}")
        except Exception as e:
            print(
                f"[FeatureExtractor] CLIP 加载失败，使用模拟特征: {e}"
            )
            self._use_real_clip = False

    def extract_text_features(self, text: str) -> np.ndarray:
        """提取文本特征向量"""
        if self._use_real_clip and self._model is not None:
            import torch

            inputs = self._processor(
                text=[text], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                text_features = self._model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            features = text_features.cpu().numpy().flatten()
            features = features / (np.linalg.norm(features) + 1e-8)
            return features

        return self._simulate_text_features(text)

    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """提取图像特征向量"""
        if self._use_real_clip and self._model is not None:
            import torch

            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = self._model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
            features = image_features.cpu().numpy().flatten()
            features = features / (np.linalg.norm(features) + 1e-8)
            return features

        return self._simulate_image_features(image)

    def _simulate_text_features(self, text: str) -> np.ndarray:
        """基于文本哈希生成确定性的模拟特征"""
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        features = rng.randn(self.feature_dim).astype(np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)
        return features

    def _simulate_image_features(self, image: Image.Image) -> np.ndarray:
        """基于图像像素哈希生成确定性的模拟特征"""
        img_small = image.resize((8, 8)).convert("RGB")
        pixels = np.array(img_small).flatten()
        seed = int(pixels.sum()) % (2**31)
        rng = np.random.RandomState(seed)
        features = rng.randn(self.feature_dim).astype(np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)
        return features

    @property
    def is_real_clip(self) -> bool:
        """是否使用了真实的 CLIP 模型"""
        return self._use_real_clip
