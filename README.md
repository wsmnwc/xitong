<div align="center">

# 🛡️ 多模态仇恨言论智能检测系统

**Multimodal Hate Speech Intelligent Detection System**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-OpenAI-412991?logo=openai)](https://openai.com/research/clip)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-3.x-orange?logo=gradio)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*面向专业硕士学位论文的图文多模态仇恨言论自动检测与可解释性分析系统*

</div>

---

## 📖 项目简介

本系统是一个基于 **CLIP 视觉-语言预训练模型** 的多模态仇恨言论检测平台，专为社交媒体内容审核场景设计。系统融合图像与文本两路模态信息，内置两种自研核心算法：

- **CF-DMW**（Counterfactual Dynamic Modal Weighting，基于动态权重的反事实去偏算法）
- **CF-DF**（Counterfactual Diffusion-based Debiasing，基于扩散净化的反事实去偏算法）

用户可通过简洁的 Web 界面上传图文样本，系统将实时输出仇恨判定结果、概率分数及可解释性可视化图表，为内容审核员提供直观的决策依据。

---

## 🎯 系统核心功能

### 模块一：多模态数据录入与预处理

| 功能 | 描述 |
|------|------|
| **单样本检测** | 上传一张图片 + 输入/OCR 识别文本，提交即得检测结果 |
| **批量检测** | 上传 CSV / JSON 文件，模拟社交平台批量内容审核场景 |
| **图像预处理** | 自动调整分辨率、归一化，适配 CLIP ViT 输入规范 |
| **文本预处理** | 分词、截断至 77 token，支持中英文混合 |
| **OCR 辅助识别** | 可选调用 OCR 引擎自动提取图中嵌入文字 |

### 模块二：智能检测与算法调度（核心）

```
用户选择算法
     │
     ├── CF-DMW ──► CLIP 特征提取 ──► 动态权重计算 (w_t, w_v) ──► 加权融合 ──► 分类头输出
     │
     └── CF-DF  ──► CLIP 特征提取 ──► 扩散净化去偏 ──► 净化特征 ──► 分类头输出
```

- **算法一：CF-DMW** — 根据当前样本自适应计算文本权重 $w_t$ 与视觉权重 $w_v$（$w_t + w_v = 1$），动态决定两路模态对最终判决的贡献比例，有效缓解模态偏差。
- **算法二：CF-DF** — 利用扩散模型生成反事实样本，将偏差特征从原始表征中"净化"去除，提升跨数据集的泛化能力。

### 模块三：结果可视化与可解释性分析

| 可视化内容 | 说明 |
|-----------|------|
| **判定标签** | 🔴 仇恨 / 🟢 安全，配以醒目颜色标注 |
| **概率分数** | 显示仇恨概率（如 `85% 危险`），含置信度进度条 |
| **模态权重图（CF-DMW）** | 饼图 / 条形图展示 $w_t$ 与 $w_v$ 占比，解释模型决策依据 |
| **净化对比图（CF-DF）** | 展示去偏前后概率分数变化，直观体现"扩散净化"效果 |
| **批量统计图** | 批量检测时输出样本分布饼图与置信度直方图 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     前端界面层                           │
│              Gradio / Streamlit Web UI                  │
│   [图文上传] [算法选择] [检测按钮] [结果可视化面板]       │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP / WebSocket
┌────────────────────────▼────────────────────────────────┐
│                     后端服务层                           │
│                  FastAPI / Flask                         │
│   [/detect 单样本] [/batch 批量] [/health 健康检查]      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                     算法推理层                           │
│                                                         │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │   CLIP Encoder  │      │      模型权重(.pth)       │  │
│  │  图像编码器      │      │  CF-DMW / CF-DF 分类头   │  │
│  │  文本编码器      │      │                         │  │
│  └────────┬────────┘      └────────────┬────────────┘  │
│           │  特征向量                   │ 推理结果        │
│           └──────────────┬─────────────┘               │
│                          │                              │
│              ┌───────────▼──────────┐                  │
│              │   算法调度器          │                  │
│              │ (cf_dmw / cf_df)     │                  │
│              └──────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 技术栈

| 层次 | 技术选型 | 用途 |
|------|---------|------|
| **前端** | Gradio 或 Streamlit | 交互界面（Python 原生，无需 JS） |
| **后端** | FastAPI + Uvicorn | RESTful API 服务，支持异步推理 |
| **视觉-语言模型** | OpenAI CLIP (ViT-B/32) | 多模态特征提取 |
| **深度学习框架** | PyTorch 2.x | 模型加载与前向推理 |
| **可视化** | Matplotlib / Plotly | 权重图、置信度图表 |
| **OCR（可选）** | PaddleOCR / EasyOCR | 图文混排场景文字提取 |
| **批量处理** | Pandas | CSV/JSON 文件解析与结果导出 |

---

## 📂 项目结构

```
xitong/
├── README.md                    # 项目说明文档
│
├── backend/                     # 后端服务
│   ├── main.py                  # FastAPI 应用入口
│   ├── api/
│   │   ├── detect.py            # 单样本检测路由
│   │   └── batch.py             # 批量检测路由
│   ├── models/
│   │   ├── cf_dmw.py            # CF-DMW 算法模型定义
│   │   └── cf_df.py             # CF-DF 算法模型定义
│   ├── utils/
│   │   ├── preprocess.py        # 图像/文本预处理
│   │   └── visualize.py         # 可视化图表生成
│   └── weights/                 # 预训练权重文件（.pth）
│       ├── cf_dmw_best.pth
│       └── cf_df_best.pth
│
├── frontend/                    # 前端界面
│   └── app.py                   # Gradio / Streamlit 主程序
│
├── data/                        # 示例数据
│   ├── sample_image.jpg
│   └── batch_sample.csv
│
├── notebooks/                   # 实验分析笔记本
│   └── visualization_demo.ipynb
│
├── requirements.txt             # Python 依赖
└── .gitignore
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.7+（推荐 GPU 推理，CPU 亦可运行）
- 显存 ≥ 6GB（使用 CLIP ViT-B/32）

### 1. 克隆仓库

```bash
git clone https://github.com/wsmnwc/xitong.git
cd xitong
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 核心依赖：

```
torch>=2.0.0
torchvision>=0.15.0
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
fastapi>=0.100.0
uvicorn[standard]
gradio>=3.40.0
matplotlib
plotly
pandas
pillow
python-multipart
```

### 3. 放置模型权重

将训练好的 `.pth` 权重文件放入 `backend/weights/` 目录：

```
backend/weights/
├── cf_dmw_best.pth    # CF-DMW 算法最优权重
└── cf_df_best.pth     # CF-DF 算法最优权重
```

### 4. 启动后端服务

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API 文档访问：[http://localhost:8000/docs](http://localhost:8000/docs)

### 5. 启动前端界面

```bash
cd frontend
python app.py
```

浏览器访问：[http://localhost:7860](http://localhost:7860)

---

## 🔌 API 接口说明

### 单样本检测

```http
POST /api/detect
Content-Type: multipart/form-data

参数:
  image  : 图片文件（JPG/PNG）
  text   : 配套文本内容
  model  : 算法选择（"cf_dmw" | "cf_df"）
```

响应示例：

```json
{
  "label": "hate",
  "probability": 0.852,
  "modal_weights": {
    "text_weight": 0.73,
    "vision_weight": 0.27
  },
  "explanation": "模型主要依赖文字信息（73%）判定此样本为仇恨言论"
}
```

### 批量检测

```http
POST /api/batch
Content-Type: multipart/form-data

参数:
  file   : CSV 或 JSON 文件
  model  : 算法选择（"cf_dmw" | "cf_df"）
```

响应：包含每条样本检测结果的 JSON 列表，支持导出为 CSV。

---

## 📊 算法详解

### CF-DMW — 基于动态权重的反事实去偏

**核心思想**：针对现有多模态模型对单一模态（如纯文字捷径）的过度依赖问题，CF-DMW 在推理阶段动态计算每个样本对应的模态权重 $(w_t, w_v)$，并将其用于融合两路 CLIP 特征，从而在无需重新训练的情况下缓解模态偏差。

$$
\hat{y} = \text{Classifier}\left(w_t \cdot \mathbf{f}_\text{text} + w_v \cdot \mathbf{f}_\text{vision}\right), \quad w_t + w_v = 1
$$

**可解释性**：权重 $w_t$、$w_v$ 直接展示给用户，说明"模型依赖文字/图像程度"。

### CF-DF — 基于扩散净化的反事实去偏

**核心思想**：利用扩散模型（Diffusion Model）对输入特征进行"反事实净化"，生成去除偏差成分的特征表示，再送入分类器进行判断。通过展示净化前后置信度的变化，用户可直观理解偏差的影响。

$$
\mathbf{f}_\text{clean} = \text{Diffusion-Purify}(\mathbf{f}_\text{original}) \quad \Rightarrow \quad \hat{y} = \text{Classifier}(\mathbf{f}_\text{clean})
$$

**可解释性**：对比净化前后的仇恨概率曲线，体现"扩散净化"对去除偏差的具体效果。

---

## 🖼️ 界面截图

> *系统运行后，在此处添加界面截图以展示实际效果。*

| 单样本检测界面 | 模态权重可视化 |
|:---:|:---:|
| *(待补充)* | *(待补充)* |

| 批量检测结果 | CF-DF 净化对比图 |
|:---:|:---:|
| *(待补充)* | *(待补充)* |

---

## 📈 实验数据集

系统在以下公开数据集上进行了训练与验证：

| 数据集 | 类型 | 样本数 | 说明 |
|-------|------|--------|------|
| **HateMM** | 图文多模态 | ~10K | 英文社交媒体帖子 |
| **MMHS150K** | 图文多模态 | 150K | Twitter 图文仇恨样本 |
| **FHM** | 图文多模态 | 10K | Facebook 仇恨梗图数据集 |

---

## 📋 论文对应章节

| 系统模块 | 对应论文章节 |
|---------|-----------|
| 多模态数据录入与预处理 | 第2章：相关工作与数据处理 |
| CF-DMW 算法 | 第3章：基于动态权重的反事实去偏方法 |
| CF-DF 算法 | 第4章：基于扩散净化的反事实去偏方法 |
| 系统设计与实现 | 第5章：多模态仇恨言论检测系统设计与实现 |
| 可视化与可解释性分析 | 第5章：系统评估与可解释性分析 |

---

## 🤝 贡献与引用

如本系统或代码对您的研究有所帮助，请引用：

```bibtex
% 请将下方 author 和 school 字段替换为您的真实姓名和所在高校
@mastersthesis{xitong2025multimodal,
  title  = {基于反事实去偏的多模态仇恨言论检测系统},
  author = {姓名},
  school = {高校名称},
  year   = {2025},
  type   = {专业学位硕士论文}
}
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

<div align="center">
  <sub>Built with ❤️ for Master's Thesis | 专业硕士学位论文系统</sub>
</div>
