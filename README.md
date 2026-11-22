# 声见其迹 SONOTRACE · 比赛展示端（前后端）

> 生理声音（心音 PCG）可视化与智能辅助分析系统 —— 互联网+ / 创新创业大赛路演演示端。
>
> **论文训练代码与实验流水线在独立目录 `paper/`，本仓库只包含比赛展示端的 Web 前后端。**
> 两者严格分离：`paper/` 负责科研可复现，`sonotrace-web/` 负责产品化演示。

本仓库所有指标数字**均由真实训练产物驱动**，不含任何硬编码的伪造结果。

---

## 1. 已接入的真实能力

| 模块 | 权重文件 | 说明 |
|---|---|---|
| 六维采集质控 | `weights/sqa_rf_model.pkl` | 论文 Stage1 随机森林，7 维特征 → 质量合格概率 |
| S1/S2 四状态分割 | `weights/seg_unet.pt` | 1D U-Net（4 通道包络）+ Viterbi 生理周期约束解码，输出 S1/S2 事件时间戳与心率周期 |
| 分类 + 域泛化 | `weights/cls_best.pt` | DualViewDGResNetCBAM（22.93 M 参数）：Log-Mel + CWT 双视图、CBAM、MixStyle、GRL；含 MC-Dropout 不确定度与 Grad-CAM |

权重缺失时后端**自动降级为 DSP 声学模式**，界面不报错、演示不中断。

---

## 2. 目录结构

```text
sonotrace-web/
├── backend/
│   ├── app.py             FastAPI：静态托管 + /api/*
│   ├── config.py          超参口径（严格对齐论文 experiment_config.yaml）
│   ├── engine.py          三项权重加载与推理（cls / seg / sqa）+ Grad-CAM
│   ├── pipeline.py        单样本端到端分析流水线
│   ├── experiments.py     论文真实指标读取层（Table 1/2/3/4/5）
│   ├── preprocess.py      信号预处理（与论文同一份实现，含 numpy2 兼容垫片）
│   └── models/            cls_models.py / seg_unet.py（与论文一致的结构定义）
├── frontend/
│   └── index.html         单文件前端（3 套主题 / 5 大模块 / 全 Canvas 渲染）
├── tools/
│   └── sync_results.py    从训练服务器同步真实产物（指标 + 权重 + 图）
├── results/               论文真实训练产物（JSON/CSV）
│   └── experiments/       每个实验的 metrics.json
├── weights/               训练权重（*.pt / *.pkl，不纳入版本库）
├── samples/               演示用心音（临床音频，不纳入版本库）
├── requirements.txt
└── .env.example
```

---

## 3. 快速启动

```bash
cd sonotrace-web
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

浏览器打开：

* 主界面 `http://127.0.0.1:8000`
* 接口文档 `http://127.0.0.1:8000/docs`

支持深链接：`#dashboard` / `#acquisition` / `#visual` / `#ai` / `#research`

> 无 GPU 亦可运行（CPU 推理单样本约 0.5 s）。有 CUDA 时自动使用 GPU。

---

## 4. 从训练服务器同步真实产物

权重与实验结果不随仓库分发，用同步工具从 3090 服务器拉取：

```bash
# 1) 复制 .env.example 为本地环境变量（不要提交真实凭据）
export SONOTRACE_SSH_HOST=1.2.3.4
export SONOTRACE_SSH_PORT=22
export SONOTRACE_SSH_USER=linux
export SONOTRACE_SSH_PASS='******'

# 2) 同步
python tools/sync_results.py                # 全量
python tools/sync_results.py --only-metrics  # 只拉指标
python tools/sync_results.py --dry-run       # 预览
```

同步完成后重启后端，`/api/health` 会显示三项权重已加载。

---

## 5. API 一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 服务、设备、三项权重与引擎模式 |
| GET | `/api/models` | 模型加载详情（含参数量、协议、推理设备） |
| GET | `/api/overview` | 首页聚合指标 |
| GET | `/api/experiments` | 全部论文实验真实指标（Table 2/3/4/5） |
| GET | `/api/experiments/{name}` | 单实验详情 |
| GET | `/api/sqa` | Stage1 六维 SQA 标定结果 |
| GET | `/api/seg` | Stage2 分割训练曲线 |
| GET | `/api/datasets` | Table 1 数据集统计 |
| GET | `/api/samples` | 内置演示样本列表 |
| GET | `/api/sample_audio/{name}` | 回放样本原始 wav（Web Audio 解码） |
| POST | `/api/analyze` | 上传 wav → 完整分析（质控 + 声像 + 分割 + 分类 + Grad-CAM） |
| POST | `/api/analyze_sample?name=` | 分析内置样本 |
| POST | `/api/quality_check` | 仅六维质控（供 ESP32 硬件前置拦截） |

---

## 6. 硬件联调约定（ESP32 数字听诊器）

* 采样格式：单声道 16-bit PCM `.wav`，建议 **4,000 Hz**（后端自动抗混叠重采样到 2 kHz）
* 时长建议 5–15 秒
* 上传：`multipart/form-data` → 先调 `/api/quality_check` 拦截低质量录音，再调 `/api/analyze`

---

## 7. 验证结果（真实样本，非演示造数）

用论文权重对内置 6 条真实临床录音推理，与清单中的临床真值对照：

| 样本 | 临床真值 | 模型输出 | 结果 |
|---|---|---|---|
| CirCor 13918_AV / MV / PV / TV | `Murmur:Present \| Outcome:Abnormal` | 异常概率 0.78 – 0.91 | 4/4 正确 |
| ZCHSound ZCH0001 / ZCH0002 | `NORMAL` | 异常概率 0.01 | 2/2 正确 |

分割模型在全部 6 条样本上均识别出规律 S1/S2 事件，周期 560–740 ms（≈ 81–107 BPM），生理范围合理。

---

## 8. 数据与合规声明

* `samples/` 与 `weights/` **不纳入版本库**：
  CirCor DigiScope 与 PhysioNet/CinC 2016 属**受控访问（credentialed access）**数据集，
  ZCHSound 为医院内部数据，将原始录音或由其派生的权重公开发布可能违反数据使用协议。
* 本系统输出**仅为辅助分析参考，不替代专业医师诊断**。界面与报告中均含此声明。

---

## 9. 与论文目录的关系

| 目录 | 职责 | 是否可公开 |
|---|---|---|
| `paper/` | 论文训练脚本、5 折防泄漏清单、消融实验流水线 | 代码可公开；manifests 中不含音频本体 |
| `sonotrace-web/` | 比赛展示端前后端（本仓库） | 可公开（权重/样本除外） |

两者共用同一份预处理口径与模型结构定义，保证"论文指标 = 演示结果"。