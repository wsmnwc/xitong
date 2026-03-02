"""
Gradio 前端界面

方面级多模态仇恨言论检测系统的交互式界面，包含三个核心模块：
1. 多模态数据录入与预处理
2. 智能检测与算法调度
3. 结果可视化与可解释性分析
"""

import io
import json
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
    parse_batch_csv,
    parse_batch_json,
    decode_base64_image,
)

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
        f"Text / Wenben\n({text_weight:.1%})",
        f"Image / Tuxiang\n({image_weight:.1%})",
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
    ax.set_title("Modal Weight Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def create_confidence_bar_chart(
    raw_prob: float, purified_prob: float
) -> plt.Figure:
    """创建去偏前后置信度对比条形图"""
    fig, ax = plt.subplots(figsize=(5, 4))
    categories = ["Before\nPurification", "After\nPurification"]
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
    ax.set_ylabel("Hate Probability", fontsize=12)
    ax.set_title("Confidence Before vs After Purification", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def detect_single(text: str, image, algorithm_display: str):
    """单样本检测核心逻辑"""
    if not text or not text.strip():
        return (
            "**Please enter text content for detection.**",
            None,
            "",
        )

    algorithm = ALGO_MAP.get(algorithm_display, ALGORITHM_CF_DMW)
    text = clean_text(text)
    extractor = get_extractor()

    if image is not None:
        pil_image = load_image(image)
    else:
        pil_image = create_placeholder_image()

    text_features = extractor.extract_text_features(text)
    image_features = extractor.extract_image_features(pil_image)

    model = get_model(algorithm)
    result = model.predict(text_features, image_features)

    if result.is_hateful:
        status_emoji = "\U0001f6a8"
        status_text = "HATE SPEECH DETECTED"
        status_color = "red"
    else:
        status_emoji = "\u2705"
        status_text = "SAFE CONTENT"
        status_color = "green"

    result_md = f"""
## {status_emoji} Detection Result: <span style="color:{status_color}">{status_text}</span>

| Item | Value |
|------|-------|
| **Hate Probability** | **{result.hate_probability:.1%}** |
| **Text Weight (w_t)** | {result.text_weight:.4f} |
| **Image Weight (w_v)** | {result.image_weight:.4f} |
| **Algorithm** | {algorithm_display} |

### Explanation
{result.explanation}
"""

    if algorithm == ALGORITHM_CF_DMW:
        chart = create_weight_pie_chart(result.text_weight, result.image_weight)
    else:
        extra = result.extra or {}
        raw_prob = extra.get("raw_probability", result.hate_probability)
        purified_prob = extra.get("purified_probability", result.hate_probability)
        chart = create_confidence_bar_chart(raw_prob, purified_prob)

    extra_json = json.dumps(result.extra, ensure_ascii=False, indent=2) if result.extra else "{}"

    return result_md, chart, extra_json


def detect_batch(file, algorithm_display: str):
    """批量检测核心逻辑"""
    if file is None:
        return "**Please upload a CSV or JSON file.**", None, None

    algorithm = ALGO_MAP.get(algorithm_display, ALGORITHM_CF_DMW)
    file_path = file.name if hasattr(file, "name") else str(file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if file_path.endswith(".csv"):
        items = parse_batch_csv(content)
    elif file_path.endswith(".json"):
        items = parse_batch_json(content)
    else:
        return "**Only .csv and .json formats are supported.**", None, None

    if not items:
        return "**No valid data entries found in the file.**", None, None

    extractor = get_extractor()
    model = get_model(algorithm)
    results_data = []

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

        results_data.append(
            {
                "Text": text[:50] + ("..." if len(text) > 50 else ""),
                "Result": "Hate" if result.is_hateful else "Safe",
                "Probability": f"{result.hate_probability:.1%}",
                "Text Weight": f"{result.text_weight:.4f}",
                "Image Weight": f"{result.image_weight:.4f}",
            }
        )

    df = pd.DataFrame(results_data)
    hateful_count = sum(1 for r in results_data if r["Result"] == "Hate")
    safe_count = len(results_data) - hateful_count

    summary_md = f"""
## Batch Detection Complete

| Statistics | Value |
|-----------|-------|
| **Total Samples** | {len(results_data)} |
| **Hate Speech** | {hateful_count} |
| **Safe Content** | {safe_count} |
| **Hate Ratio** | {hateful_count / len(results_data):.1%} |
| **Algorithm** | {algorithm_display} |
"""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Hate Speech", "Safe Content"],
        [hateful_count, safe_count],
        color=["#FF6B6B", "#5AD8A6"],
        width=0.5,
        edgecolor="white",
    )
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Batch Detection Results", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (label, count) in enumerate(
        zip(["Hate", "Safe"], [hateful_count, safe_count])
    ):
        ax.text(i, count + 0.2, str(count), ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return summary_md, df, fig


def build_interface() -> gr.Blocks:
    """构建 Gradio 界面"""
    _theme = gr.themes.Soft(primary_hue="blue")
    _css = """
    .result-hateful { background-color: #fee2e2 !important; border: 2px solid #ef4444 !important; }
    .result-safe { background-color: #dcfce7 !important; border: 2px solid #22c55e !important; }
    """
    with gr.Blocks(
        title="Aspect-Level Multimodal Hate Speech Detection System",
    ) as demo:
        demo.theme = _theme
        demo.css = _css
        gr.Markdown(
            """
        # \U0001f6e1\ufe0f Aspect-Level Multimodal Hate Speech Detection System
        ### Fangmianji Duomotai Chouhen Yanlun Jiance Xitong

        This system supports two detection algorithms:
        - **CF-DMW**: Counterfactual detection based on dynamic modal weights — visualizes the contribution of text vs image modalities
        - **CF-DF**: Counterfactual detection based on diffusion purification — shows confidence before and after debiasing
        """
        )

        with gr.Tabs():
            # === Tab 1: Single sample detection ===
            with gr.TabItem("\U0001f50d Single Sample Detection"):
                gr.Markdown("### Upload an image and enter text for detection")
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload Image (optional)",
                            type="pil",
                        )
                        input_text = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter the text content to detect...",
                            lines=3,
                        )
                        algo_select = gr.Dropdown(
                            choices=list(ALGO_MAP.keys()),
                            value=list(ALGO_MAP.keys())[0],
                            label="Select Algorithm",
                        )
                        detect_btn = gr.Button(
                            "\U0001f680 Start Detection",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        result_output = gr.Markdown(label="Detection Result")
                        chart_output = gr.Plot(label="Visualization")
                        extra_output = gr.Code(
                            label="Detailed Data (JSON)",
                            language="json",
                        )

                detect_btn.click(
                    fn=detect_single,
                    inputs=[input_text, input_image, algo_select],
                    outputs=[result_output, chart_output, extra_output],
                )

                gr.Markdown("---")
                gr.Markdown("### \U0001f4a1 Quick Test Examples")
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

            # === Tab 2: Batch detection ===
            with gr.TabItem("\U0001f4c1 Batch Detection"):
                gr.Markdown(
                    """### Upload a CSV or JSON file for batch detection
CSV format: must include a `text` column, and optionally an `image` column (Base64 encoded).
JSON format: an array of objects, each containing `text` and optionally `image` fields."""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_file = gr.File(
                            label="Upload File (.csv or .json)",
                            file_types=[".csv", ".json"],
                        )
                        batch_algo = gr.Dropdown(
                            choices=list(ALGO_MAP.keys()),
                            value=list(ALGO_MAP.keys())[0],
                            label="Select Algorithm",
                        )
                        batch_btn = gr.Button(
                            "\U0001f680 Start Batch Detection",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        batch_summary = gr.Markdown(label="Batch Summary")
                        batch_table = gr.Dataframe(label="Detailed Results")
                        batch_chart = gr.Plot(label="Result Statistics")

                batch_btn.click(
                    fn=detect_batch,
                    inputs=[batch_file, batch_algo],
                    outputs=[batch_summary, batch_table, batch_chart],
                )

            # === Tab 3: System info ===
            with gr.TabItem("\u2139\ufe0f System Info"):
                gr.Markdown(
                    f"""
### System Configuration

| Item | Detail |
|------|--------|
| **CLIP Model** | `openai/clip-vit-base-patch32` |
| **Device** | `{DEVICE}` |
| **Algorithm 1** | CF-DMW (Counterfactual Dynamic Modal Weights) |
| **Algorithm 2** | CF-DF (Counterfactual Diffusion Purification) |
| **Framework** | PyTorch + FastAPI + Gradio |

### Algorithm Descriptions

#### CF-DMW: Counterfactual Detection Based on Dynamic Modal Weights
This algorithm dynamically computes the text modal weight $w_t$ and image modal weight $w_v$,
adaptively assigning contribution from each modality for different samples,
thus improving detection performance.

#### CF-DF: Counterfactual Detection Based on Diffusion Purification
This algorithm uses a diffusion process to debias and purify features,
eliminating spurious correlations. It compares confidence before and after purification
to demonstrate the debiasing effect.

### Usage Guide
1. **Single Detection**: Upload an image and enter text, select an algorithm, click "Start Detection"
2. **Batch Detection**: Prepare a CSV/JSON file, upload and select an algorithm, click "Start Batch Detection"
3. **Result Interpretation**: Green = safe, Red = hate speech; check the visualization charts for explainability
"""
                )

    return demo
