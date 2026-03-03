"""
OCR 文字识别模块

从图片中提取英文文本。
"""

from PIL import Image

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'])
    return _reader


def extract_text_from_image(image: Image.Image) -> str:
    """
    使用 OCR 从图片中提取英文文本。

    优先使用 easyocr（Colab 上容易安装），
    如果不可用则返回提示信息。
    """
    try:
        import numpy as np
        reader = _get_reader()
        results = reader.readtext(np.array(image))
        text = ' '.join([r[1] for r in results])
        return text
    except ImportError:
        return "[OCR 未安装] 请运行: pip install easyocr"
