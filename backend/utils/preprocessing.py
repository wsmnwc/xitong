"""
数据预处理模块

处理图像加载、文本清洗、批量数据解析等。
"""

import csv
import io
import json
import base64

import numpy as np
from PIL import Image


def load_image(image_input) -> Image.Image:
    """
    从多种输入加载 PIL 图像。

    Args:
        image_input: 文件路径(str)、PIL.Image、numpy数组或bytes

    Returns:
        PIL.Image.Image: RGB 格式的图像
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    raise ValueError(f"不支持的图像输入类型: {type(image_input)}")


def clean_text(text: str) -> str:
    """
    清洗输入文本

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text:
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text


def parse_batch_csv(file_content: str | bytes) -> list[dict]:
    """
    解析批量检测的 CSV 文件。

    CSV 格式要求：
    - text 列：文本内容
    - image 列（可选）：图像的 Base64 编码或文件路径

    Args:
        file_content: CSV 文件的字符串或字节内容

    Returns:
        包含 {"text": str, "image_base64": str | None} 的列表
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    reader = csv.DictReader(io.StringIO(file_content))
    results = []
    for row in reader:
        item = {
            "text": clean_text(row.get("text", "")),
            "image_base64": row.get("image", None),
        }
        if item["text"]:
            results.append(item)
    return results


def parse_batch_json(file_content: str | bytes) -> list[dict]:
    """
    解析批量检测的 JSON 文件。

    JSON 格式要求：
    [
        {"text": "...", "image": "base64 或路径"},
        ...
    ]

    Args:
        file_content: JSON 文件的字符串或字节内容

    Returns:
        包含 {"text": str, "image_base64": str | None} 的列表
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    data = json.loads(file_content)
    if not isinstance(data, list):
        raise ValueError("JSON 文件必须是一个数组")

    results = []
    for item in data:
        entry = {
            "text": clean_text(item.get("text", "")),
            "image_base64": item.get("image", None),
        }
        if entry["text"]:
            results.append(entry)
    return results


def is_mami_format(file_content: str | bytes) -> bool:
    """
    检查 CSV/TSV 内容是否是 MAMI 数据集格式。

    MAMI 格式的第一行（标题行）必须同时包含 'file_name' 和
    'Text Transcription' 列名。

    Args:
        file_content: CSV/TSV 文件的字符串或字节内容

    Returns:
        True 如果是 MAMI 格式，否则 False
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")
    first_line = file_content.split("\n")[0]
    return "file_name" in first_line and "Text Transcription" in first_line


def parse_mami_csv(file_content: str | bytes, image_files: dict | None = None) -> list[dict]:
    """
    解析 MAMI 数据集格式的 TSV/CSV 文件。

    MAMI 格式：tab 分隔，包含 file_name 和 Text Transcription 列。

    Args:
        file_content: TSV/CSV 文件内容
        image_files: 文件名到图片字节的映射 {filename: bytes}

    Returns:
        包含 {"text": str, "image_base64": str | None, "file_name": str} 的列表
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    reader = csv.DictReader(io.StringIO(file_content), delimiter="\t")
    results = []
    for row in reader:
        file_name = row.get("file_name", "").strip()
        text = clean_text(row.get("Text Transcription", ""))
        image_base64 = None
        if image_files and file_name in image_files:
            image_bytes = image_files[file_name]
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        if text:
            results.append({
                "text": text,
                "image_base64": image_base64,
                "file_name": file_name,
            })
    return results


def decode_base64_image(base64_str: str) -> Image.Image:
    """解码 Base64 编码的图像"""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def create_placeholder_image(width: int = 224, height: int = 224) -> Image.Image:
    """创建占位图像（用于仅有文本输入的场景）"""
    return Image.new("RGB", (width, height), color=(128, 128, 128))
