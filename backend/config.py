"""系统全局配置"""

import os

# 服务配置
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "7860"))

# 模型配置
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.getenv("DEVICE", "cpu")

# CF-DMW 模型权重路径
CF_DMW_WEIGHTS_PATH = os.getenv("CF_DMW_WEIGHTS_PATH", "")
# CF-DF 模型权重路径
CF_DF_WEIGHTS_PATH = os.getenv("CF_DF_WEIGHTS_PATH", "")

# CLIP 特征维度
CLIP_FEATURE_DIM = 512

# 分类阈值
HATE_THRESHOLD = 0.5

# 算法名称
ALGORITHM_CF_DMW = "cf-dmw"
ALGORITHM_CF_DF = "cf-df"
ALGORITHM_CHOICES = [ALGORITHM_CF_DMW, ALGORITHM_CF_DF]
