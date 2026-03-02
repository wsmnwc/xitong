"""
FastAPI 后端应用

提供 REST API 用于多模态仇恨言论检测。
"""

import base64
import io
import traceback

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import (
    ALGORITHM_CF_DF,
    ALGORITHM_CF_DMW,
    ALGORITHM_CHOICES,
    DEVICE,
)
from backend.models.cf_df import CfDfModel
from backend.models.cf_dmw import CfDmwModel
from backend.utils.feature_extraction import FeatureExtractor
from backend.utils.preprocessing import (
    clean_text,
    create_placeholder_image,
    decode_base64_image,
    load_image,
    parse_batch_csv,
    parse_batch_json,
)

app = FastAPI(
    title="多模态仇恨言论检测系统",
    description="基于 CLIP 的多模态仇恨言论检测 API，支持 CF-DMW 和 CF-DF 两种算法",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例（延迟初始化）
_models: dict = {}
_extractor: FeatureExtractor | None = None


def get_extractor() -> FeatureExtractor:
    """获取特征提取器单例"""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def get_model(algorithm: str):
    """获取对应算法的模型单例"""
    if algorithm not in _models:
        if algorithm == ALGORITHM_CF_DMW:
            _models[algorithm] = CfDmwModel(device=DEVICE)
        elif algorithm == ALGORITHM_CF_DF:
            _models[algorithm] = CfDfModel(device=DEVICE)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
    return _models[algorithm]


class DetectionRequest(BaseModel):
    """单样本检测请求"""

    text: str
    image_base64: str | None = None
    algorithm: str = ALGORITHM_CF_DMW


class DetectionResponse(BaseModel):
    """检测响应"""

    is_hateful: bool
    hate_probability: float
    text_weight: float
    image_weight: float
    explanation: str
    extra: dict | None = None


class BatchDetectionResponse(BaseModel):
    """批量检测响应"""

    results: list[DetectionResponse]
    total: int
    hateful_count: int
    safe_count: int


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    clip_loaded: bool
    available_algorithms: list[str]


@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查接口"""
    extractor = get_extractor()
    return HealthResponse(
        status="ok",
        clip_loaded=extractor.is_real_clip,
        available_algorithms=ALGORITHM_CHOICES,
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_single(
    text: str = Form(...),
    algorithm: str = Form(ALGORITHM_CF_DMW),
    image: UploadFile | None = File(None),
):
    """
    单样本检测接口

    接受表单形式的文本和可选图片文件上传。
    """
    text = clean_text(text)
    extractor = get_extractor()

    if image is not None:
        image_bytes = await image.read()
        pil_image = load_image(image_bytes)
    else:
        pil_image = create_placeholder_image()

    text_features = extractor.extract_text_features(text)
    image_features = extractor.extract_image_features(pil_image)

    model = get_model(algorithm)
    result = model.predict(text_features, image_features)

    return DetectionResponse(
        is_hateful=result.is_hateful,
        hate_probability=result.hate_probability,
        text_weight=result.text_weight,
        image_weight=result.image_weight,
        explanation=result.explanation,
        extra=result.extra,
    )


@app.post("/detect/json", response_model=DetectionResponse)
async def detect_single_json(request: DetectionRequest):
    """
    单样本检测接口（JSON 格式）

    接受 JSON 格式的请求体。
    """
    text = clean_text(request.text)
    extractor = get_extractor()

    if request.image_base64:
        pil_image = decode_base64_image(request.image_base64)
    else:
        pil_image = create_placeholder_image()

    text_features = extractor.extract_text_features(text)
    image_features = extractor.extract_image_features(pil_image)

    model = get_model(request.algorithm)
    result = model.predict(text_features, image_features)

    return DetectionResponse(
        is_hateful=result.is_hateful,
        hate_probability=result.hate_probability,
        text_weight=result.text_weight,
        image_weight=result.image_weight,
        explanation=result.explanation,
        extra=result.extra,
    )


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    file: UploadFile = File(...),
    algorithm: str = Form(ALGORITHM_CF_DMW),
):
    """
    批量检测接口

    支持上传 CSV 或 JSON 文件进行批量检测。
    """
    content = await file.read()
    filename = file.filename or ""

    if filename.endswith(".csv"):
        items = parse_batch_csv(content)
    elif filename.endswith(".json"):
        items = parse_batch_json(content)
    else:
        raise ValueError("仅支持 .csv 和 .json 格式的文件")

    extractor = get_extractor()
    model = get_model(algorithm)
    results = []

    for item in items:
        text = item["text"]
        if item.get("image_base64"):
            try:
                pil_image = decode_base64_image(item["image_base64"])
            except Exception:
                pil_image = create_placeholder_image()
        else:
            pil_image = create_placeholder_image()

        text_features = extractor.extract_text_features(text)
        image_features = extractor.extract_image_features(pil_image)
        result = model.predict(text_features, image_features)

        results.append(
            DetectionResponse(
                is_hateful=result.is_hateful,
                hate_probability=result.hate_probability,
                text_weight=result.text_weight,
                image_weight=result.image_weight,
                explanation=result.explanation,
                extra=result.extra,
            )
        )

    hateful_count = sum(1 for r in results if r.is_hateful)

    return BatchDetectionResponse(
        results=results,
        total=len(results),
        hateful_count=hateful_count,
        safe_count=len(results) - hateful_count,
    )
