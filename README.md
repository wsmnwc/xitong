# 🛡️ 方面级多模态仇恨言论检测系统

Aspect-Level Multimodal Hate Speech Detection System

基于 CLIP 的多模态仇恨言论检测系统，支持两种反事实检测算法（CF-DMW 和 CF-DF），提供完整的前后端交互和可解释性可视化。

## 系统架构

- **前端**：Gradio（Python 交互式 Web 界面）
- **后端**：FastAPI（REST API 服务）
- **算法**：PyTorch + CLIP 预训练模型
- **可视化**：Matplotlib（模态权重饼图、置信度对比条形图）

## 核心功能模块

### 模块一：多模态数据录入与预处理
- 单样本检测：上传图片 + 输入文本
- 批量检测：支持 CSV / JSON 文件上传

### 模块二：智能检测与算法调度
- **CF-DMW**：基于动态模态权重的反事实检测 — 自适应计算文本/图像模态权重
- **CF-DF**：基于扩散净化的反事实检测 — 通过扩散过程去偏，消除虚假关联

### 模块三：结果可视化与可解释性分析
- 红色（仇恨）/ 绿色（安全）标签 + 概率分数
- CF-DMW：模态权重饼图（w_t / w_v 占比）
- CF-DF：去偏前后置信度对比条形图

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动系统

```bash
# 启动 Gradio 前端界面（默认模式）
python run.py gradio

# 启动 FastAPI 后端 API
python run.py api

# 同时启动前端和后端
python run.py both
```

### 运行测试

```bash
python -m pytest tests/ -v
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/detect` | POST | 单样本检测（表单上传） |
| `/detect/json` | POST | 单样本检测（JSON） |
| `/detect/batch` | POST | 批量检测（CSV/JSON 文件） |

## 项目结构

```
├── run.py                    # 主入口
├── requirements.txt          # 依赖
├── backend/
│   ├── app.py                # FastAPI 应用
│   ├── config.py             # 全局配置
│   ├── models/
│   │   ├── base.py           # 模型基类
│   │   ├── cf_dmw.py         # CF-DMW 模型
│   │   └── cf_df.py          # CF-DF 模型
│   └── utils/
│       ├── feature_extraction.py  # CLIP 特征提取
│       └── preprocessing.py       # 数据预处理
├── frontend/
│   └── app.py                # Gradio 前端界面
└── tests/
    └── test_backend.py       # 单元测试
```

## 模型权重

系统支持加载自定义训练的 `.pth` 权重文件。通过环境变量配置路径：

```bash
export CF_DMW_WEIGHTS_PATH=/path/to/cf_dmw.pth
export CF_DF_WEIGHTS_PATH=/path/to/cf_df.pth
```

未配置权重时，系统使用默认初始化参数运行（适合演示）。