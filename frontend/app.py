"""
Gradio 前端界面

多模态仇恨言论检测系统的交互式界面，包含三个核心模块：
1. 多模态数据录入与预处理
2. 智能检测与算法调度
3. 结果可视化与可解释性分析
"""

import io
import json
import os
import tempfile

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from backend.config import ALGORITHM_CF_DF, ALGORITHM_CF_DMW, DEVICE
from backend.models.cf_df import CfDfModel
from backend.models.cf_dmw import CfDmwModel
from backend.utils.feature_extraction import FeatureExtractor
from backend.utils.preprocessing import (
    clean_text,
    create_placeholder_image,
    load_image,
    parse_batch_json,
    parse_mami_csv,
    decode_base64_image,
)
from backend.utils.ocr import extract_text_from_image

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = [
    "SimHei", "DejaVu Sans", "Arial Unicode MS", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False

# 全局实例
_extractor: FeatureExtractor | None = None
_models: dict = {}


def get_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


def get_model(algorithm: str):
    if algorithm not in _models:
        if algorithm == ALGORITHM_CF_DMW:
            _models[algorithm] = CfDmwModel(device=DEVICE)
        elif algorithm == ALGORITHM_CF_DF:
            _models[algorithm] = CfDfModel(device=DEVICE)
    return _models[algorithm]


ALGO_MAP = {
    "CF-DMW（基于动态模态权重）": ALGORITHM_CF_DMW,
    "CF-DF（基于扩散净化）": ALGORITHM_CF_DF,
}


def create_weight_pie_chart(text_weight: float, image_weight: float) -> plt.Figure:
    """创建模态权重饼图"""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [
        f"文本模态\n({text_weight:.1%})",
        f"图像模态\n({image_weight:.1%})",
    ]
    sizes = [text_weight, image_weight]
    colors = ["#5B8FF9", "#5AD8A6"]
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")
    ax.set_title("模态权重分布", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def create_text_only_chart() -> plt.Figure:
    """创建纯文本模式说明图表（无图片输入时使用）"""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [1.0],
        labels=["文本模态\n(100.0%)"],
        colors=["#5B8FF9"],
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title("模态权重分布（纯文本模式）", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.02,
        "本次检测未提供图片，仅使用文本模态",
        ha="center", fontsize=10, color="gray",
    )
    fig.tight_layout()
    return fig


def create_confidence_bar_chart(
    raw_prob: float, purified_prob: float
) -> plt.Figure:
    """创建去偏前后置信度对比条形图"""
    fig, ax = plt.subplots(figsize=(5, 4))
    categories = ["净化前", "净化后"]
    values = [raw_prob, purified_prob]
    colors = ["#FF6B6B", "#5AD8A6"]
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("仇恨概率", fontsize=12)
    ax.set_title("去偏前后置信度对比", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def detect_single(text: str, image, algorithm_display: str):
    """单样本检测核心逻辑"""
    if not text or not text.strip():
        return (
            "**请输入待检测的文本内容。**",
            None,
            "",
        )

    algorithm = ALGO_MAP.get(algorithm_display, ALGORITHM_CF_DMW)
    text = clean_text(text)
    extractor = get_extractor()

    has_image = image is not None
    if has_image:
        pil_image = load_image(image)
    else:
        pil_image = create_placeholder_image()

    text_features = extractor.extract_text_features(text)
    image_features = extractor.extract_image_features(pil_image)

    model = get_model(algorithm)
    result = model.predict(text_features, image_features, has_image=has_image)

    if result.is_hateful:
        status_emoji = "\U0001f6a8"
        status_text = "仇恨言论"
        status_color = "red"
    else:
        status_emoji = "\u2705"
        status_text = "安全内容"
        status_color = "green"

    image_weight_display = f"{result.image_weight:.4f}" if has_image else "N/A（未提供图片）"

    result_md = f"""
## {status_emoji} 检测结果：<span style="color:{status_color}">{status_text}</span>

| 项目 | 数值 |
|------|-------|
| **仇恨概率** | **{result.hate_probability:.1%}** |
| **文本权重 (w_t)** | {result.text_weight:.4f} |
| **图像权重 (w_v)** | {image_weight_display} |
| **使用算法** | {algorithm_display} |

### 📊 可解释性分析
{result.explanation}
"""

    if algorithm == ALGORITHM_CF_DMW:
        if has_image:
            chart = create_weight_pie_chart(result.text_weight, result.image_weight)
        else:
            chart = create_text_only_chart()
    else:
        extra = result.extra or {}
        raw_prob = extra.get("raw_probability", result.hate_probability)
        purified_prob = extra.get("purified_probability", result.hate_probability)
        chart = create_confidence_bar_chart(raw_prob, purified_prob)

    extra_json = json.dumps(result.extra, ensure_ascii=False, indent=2) if result.extra else "{}"

    return result_md, chart, extra_json


def detect_batch(file, image_files_upload, algorithm_display: str):
    """批量检测核心逻辑"""
    if file is None:
        return "**请上传 CSV 或 JSON 文件。**", None, None

    algorithm = ALGO_MAP.get(algorithm_display, ALGORITHM_CF_DMW)
    file_path = file.name if hasattr(file, "name") else str(file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 构建文件名 → 字节映射（用于 MAMI 格式匹配图片）
    image_file_map: dict[str, bytes] | None = None
    if image_files_upload:
        image_file_map = {}
        for img_file in image_files_upload:
            img_path = img_file.name if hasattr(img_file, "name") else str(img_file)
            fname = os.path.basename(img_path)
            with open(img_path, "rb") as f:
                image_file_map[fname] = f.read()

    if file_path.endswith(".json"):
        raw_items = parse_batch_json(content)
        items = [
            {**item, "label": None, "file_name": None}
            if "label" not in item else item
            for item in raw_items
        ]
    else:
        # CSV / TSV：自动判断 MAMI vs 简单格式
        items = parse_mami_csv(content, image_file_map)

    if not items:
        return "**文件中未找到有效数据条目。**", None, None

    extractor = get_extractor()
    model = get_model(algorithm)
    results_data = []
    has_labels = any(item.get("label") is not None for item in items)

    for item in items:
        text = item["text"]
        has_image = bool(item.get("image_base64"))
        if has_image:
            try:
                pil_image = decode_base64_image(item["image_base64"])
            except Exception:
                pil_image = create_placeholder_image()
                has_image = False
        else:
            pil_image = create_placeholder_image()

        text_features = extractor.extract_text_features(text)
        image_features = extractor.extract_image_features(pil_image)
        result = model.predict(text_features, image_features, has_image=has_image)

        row = {
            "文本": text[:50] + ("..." if len(text) > 50 else ""),
            "预测结果": "仇恨" if result.is_hateful else "安全",
            "预测概率": f"{result.hate_probability:.1%}",
            "文本权重": f"{result.text_weight:.4f}",
            "图像权重": f"{result.image_weight:.4f}" if has_image else "N/A",
        }
        if has_labels:
            label = item.get("label")
            if label is not None:
                true_label = "仇恨" if label == 1 else "安全"
                row["真实标签"] = true_label
                row["是否正确"] = "✓" if (result.is_hateful == (label == 1)) else "✗"
            else:
                row["真实标签"] = ""
                row["是否正确"] = ""
        results_data.append(row)

    df = pd.DataFrame(results_data)
    hateful_count = sum(1 for r in results_data if r["预测结果"] == "仇恨")
    safe_count = len(results_data) - hateful_count

    # 准确率统计（当有真实标签时）
    accuracy_md = ""
    if has_labels:
        labeled = [
            (r, item)
            for r, item in zip(results_data, items)
            if item.get("label") is not None
        ]
        if labeled:
            correct = sum(
                1 for r, item in labeled
                if (r["预测结果"] == "仇恨") == (item["label"] == 1)
            )
            tp = sum(
                1 for r, item in labeled
                if r["预测结果"] == "仇恨" and item["label"] == 1
            )
            fp = sum(
                1 for r, item in labeled
                if r["预测结果"] == "仇恨" and item["label"] == 0
            )
            fn = sum(
                1 for r, item in labeled
                if r["预测结果"] == "安全" and item["label"] == 1
            )
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            accuracy = correct / len(labeled)
            accuracy_md = f"""
| **准确率** | {accuracy:.1%} |
| **精确率** | {precision:.1%} |
| **召回率** | {recall:.1%} |"""

    summary_md = f"""
## 批量检测完成

| 统计项 | 数值 |
|-----------|-------|
| **总样本数** | {len(results_data)} |
| **仇恨言论** | {hateful_count} |
| **安全内容** | {safe_count} |
| **仇恨比例** | {hateful_count / len(results_data):.1%} |
| **使用算法** | {algorithm_display} |{accuracy_md}
"""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["仇恨言论", "安全内容"],
        [hateful_count, safe_count],
        color=["#FF6B6B", "#5AD8A6"],
        width=0.5,
        edgecolor="white",
    )
    ax.set_ylabel("数量", fontsize=12)
    ax.set_title("批量检测结果统计", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (label, count) in enumerate(
        zip(["仇恨言论", "安全内容"], [hateful_count, safe_count])
    ):
        ax.text(i, count + 0.2, str(count), ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return summary_md, df, fig


def build_interface() -> gr.Blocks:
    """构建 Gradio 界面"""
    _theme = gr.themes.Soft(primary_hue="blue")
    _css = """
    body, .gradio-container { font-family: "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif !important; }
    .result-hateful { background-color: #fee2e2 !important; border: 2px solid #ef4444 !important; border-radius: 8px !important; }
    .result-safe { background-color: #dcfce7 !important; border: 2px solid #22c55e !important; border-radius: 8px !important; }
    .gr-button.primary { background: linear-gradient(135deg, #1e3a5f, #2563eb) !important; font-weight: bold !important; letter-spacing: 1px !important; }
    #header-block { background: linear-gradient(135deg, #0f172a, #1e3a5f); border-radius: 12px; padding: 20px; margin-bottom: 12px; color: white !important; }
    #header-block * { color: white !important; }
    .tab-nav button { font-weight: bold !important; }
    """
    with gr.Blocks(
        title="多模态仇恨言论检测系统",
    ) as demo:
        demo.theme = _theme
        demo.css = _css
        gr.Markdown(
            """
        # 🛡️ 多模态仇恨言论检测系统
        ### 基于 CLIP + MoRE 的智能检测平台 | 专业硕士学位论文演示系统

        本系统支持两种检测算法：
        - **CF-DMW：基于动态模态权重的反事实检测** — 可视化文本与图像模态的贡献比例
        - **CF-DF：基于扩散净化的反事实检测** — 展示去偏前后的置信度变化
        """,
            elem_id="header-block",
        )

        with gr.Tabs():
            # === Tab 1: 单样本检测 ===
            with gr.TabItem("🔍 单样本检测"):
                gr.Markdown("### 上传图片并输入文本进行检测")
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="上传图片（可选）",
                            type="pil",
                        )
                        ocr_btn = gr.Button("🔍 OCR 识别文本", size="sm")
                        input_text = gr.Textbox(
                            label="输入文本",
                            placeholder="请输入待检测的英文文本内容...",
                            lines=3,
                        )
                        algo_select = gr.Dropdown(
                            choices=list(ALGO_MAP.keys()),
                            value=list(ALGO_MAP.keys())[0],
                            label="选择检测算法",
                        )
                        detect_btn = gr.Button(
                            "🚀 开始检测",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        result_output = gr.Markdown(label="检测结果")
                        chart_output = gr.Plot(label="可视化分析")
                        extra_output = gr.Code(
                            label="详细数据（JSON）",
                            language="json",
                        )

                detect_btn.click(
                    fn=detect_single,
                    inputs=[input_text, input_image, algo_select],
                    outputs=[result_output, chart_output, extra_output],
                )
                ocr_btn.click(
                    fn=extract_text_from_image,
                    inputs=[input_image],
                    outputs=[input_text],
                )

                gr.Markdown("---")
                gr.Markdown("### 💡 快速测试示例")
                gr.Examples(
                    examples=[
                        [
                            "I love this beautiful sunset photo, nature is amazing!",
                            None,
                            "CF-DMW（基于动态模态权重）",
                        ],
                        [
                            "These people are disgusting and should be eliminated",
                            None,
                            "CF-DMW（基于动态模态权重）",
                        ],
                        [
                            "What a wonderful day to enjoy life",
                            None,
                            "CF-DF（基于扩散净化）",
                        ],
                    ],
                    inputs=[input_text, input_image, algo_select],
                )

            # === Tab 2: 批量检测 ===
            with gr.TabItem("📁 批量检测"):
                gr.Markdown(
                    """### 上传 CSV、TSV 或 JSON 文件进行批量检测

支持以下三种格式：
- **格式一：MAMI 数据集格式**（TSV/CSV，tab 分隔）— 包含 `file_name` 和 `Text Transcription` 列，可同时上传对应图片文件
- **格式二：简单 CSV** — 包含 `text` 列，可选 `image` 列（Base64 编码）
- **格式三：JSON 数组** — 每个对象包含 `text` 字段，可选包含 `image` 字段"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_file = gr.File(
                            label="上传文件（.csv、.tsv 或 .json）",
                            file_types=[".csv", ".tsv", ".json"],
                        )
                        batch_images = gr.File(
                            label="上传图片文件（可选，支持多文件，用于 MAMI 格式）",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        batch_algo = gr.Dropdown(
                            choices=list(ALGO_MAP.keys()),
                            value=list(ALGO_MAP.keys())[0],
                            label="选择检测算法",
                        )
                        batch_btn = gr.Button(
                            "🚀 开始批量检测",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        batch_summary = gr.Markdown(label="批量检测摘要")
                        batch_table = gr.Dataframe(label="详细结果")
                        batch_chart = gr.Plot(label="结果统计")

                batch_btn.click(
                    fn=detect_batch,
                    inputs=[batch_file, batch_images, batch_algo],
                    outputs=[batch_summary, batch_table, batch_chart],
                )

            # === Tab 3: 系统信息 ===
            with gr.TabItem("ℹ️ 系统信息"):
                gr.Markdown(
                    f"""
### 系统配置

| 项目 | 详情 |
|------|------|
| **CLIP 模型** | `openai/clip-vit-base-patch32` |
| **运行设备** | `{DEVICE}` |
| **算法一** | CF-DMW（基于动态模态权重的反事实检测） |
| **算法二** | CF-DF（基于扩散净化的反事实检测） |
| **技术栈** | PyTorch + FastAPI + Gradio |

### 算法说明

#### CF-DMW：基于动态模态权重的反事实检测
该算法动态计算文本模态权重 $w_t$ 和图像模态权重 $w_v$，
自适应地为不同样本分配各模态的贡献度，从而提升检测性能。

#### CF-DF：基于扩散净化的反事实检测
该算法通过扩散过程对特征进行去偏净化，消除虚假关联。
通过对比净化前后的置信度变化，展示去偏效果。

### 使用指南
1. **单样本检测**：上传图片并输入文本，选择算法，点击"开始检测"
2. **批量检测**：准备 CSV/JSON 文件，上传并选择算法，点击"开始批量检测"
3. **结果解读**：绿色 = 安全，红色 = 仇恨言论；查看可视化图表了解模型判断依据
"""
                )

    return demo
