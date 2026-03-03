"""
OCR 工具模块

从图像中提取文本，优先使用 easyocr，回退到 pytesseract。
"""

import numpy as np
from PIL import Image

_easyocr_reader = None


def _get_easyocr_reader():
    """懒加载并缓存 easyocr.Reader 实例。"""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # type: ignore
        _easyocr_reader = easyocr.Reader(["en"], verbose=False)
    return _easyocr_reader


def extract_text_from_image(image: Image.Image) -> str:
    """
    使用 OCR 从图片中提取文本。

    优先使用 easyocr（Colab 上容易安装），
    如果不可用则尝试 pytesseract，
    如果都不可用则返回提示信息。

    Args:
        image: PIL.Image.Image 对象

    Returns:
        识别出的文本字符串
    """
    try:
        reader = _get_easyocr_reader()
        results = reader.readtext(np.array(image))
        text = " ".join(r[1] for r in results)
        return text
    except ImportError:
        pass

    try:
        import pytesseract  # type: ignore
        text = pytesseract.image_to_string(image)
        return text.strip()
    except ImportError:
        pass

    return "[OCR 未安装] 请运行: pip install easyocr"
