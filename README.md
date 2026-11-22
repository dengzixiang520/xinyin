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
| 六维采集质控 | `weights/sqa_rf_model.pkl` | 论文 Stage1 随机森林，6 维特征 → 质量合格概率 |
| S1/S2 四状态分割 | `weights/seg_unet.pt` | 1D U-Net（4 通道包络）+ Viterbi 生理周期约束解码，输出 S1/S2 事件时间戳与心率周期 |
| 分类 + 域泛化 | `weights/cls_best.pt` | DualViewDGResNetCBAM（22.93 M 参数）：Log-Mel + CWT 双视图、CBAM、MixStyle、GRL；含 MC-Dropout 不确定度与 Grad-CAM |
| **选择性拒识（双门控）** | — | 不确定度门 + 决策边际门，阈值在真实外测集上反解；命中即输出「建议人工复核」而不是硬判 |
| **风险-覆盖率分析** | `results/predictions/*.csv` | 覆盖率-准确率曲线、可靠度图（ECE）、跨设备域分解，全部逐条重算 |
| **多模型对比** | `weights/cls_{naive,lodo,nocwt}.pt` | 同一段录音在 4 个论文检查点上并排推理，暴露域泛化机制的真实差异 |
| **批量分析与复核队列** | `data/sonotrace.db` | 一次最多 12 条；结果落库，拒识记录自动生成待复核工单 |

权重缺失时后端**自动降级为 DSP 声学模式**，界面不报错、演示不中断。

---

## 2. 目录结构

```text
sonotrace-web/
├── backend/
│   ├── app.py             FastAPI：静态托管 + /api/*（30 个接口）
│   ├── config.py          超参口径（严格对齐论文 experiment_config.yaml）+ 检查点池
│   ├── engine.py          检查点池/分割/SQA 加载与推理 + Grad-CAM（MC-Dropout 固定种子）
│   ├── pipeline.py        单样本端到端分析 + 双门控判定 + 多模型对比
│   ├── selective.py       风险-覆盖率 / 校准 / 跨域分解（阈值反解的唯一来源）
│   ├── store.py           分析历史与复核队列（SQLite，标准库实现）
│   ├── experiments.py     论文真实指标读取层（Table 1/2/3/4/5 + 消融差值 + 数据审计）
│   ├── preprocess.py      信号预处理（与论文同一份实现，含 numpy2 兼容垫片）
│   └── models/            cls_models.py / seg_unet.py（与论文一致的结构定义）
├── frontend/
│   └── index.html         单文件前端（3 套主题 / 6 大模块 / 全 Canvas 渲染）
├── tools/
│   ├── sync_results.py    从训练服务器同步真实产物（指标 + 权重 + 图 + 逐条预测）
│   └── export_tables.py   把真实产物导出成 docs/tables/*.csv，供评委直接取用
├── docs/
│   ├── 竞赛答辩报告.md     ★ 答辩主文档：数据 / 提升 / 边界 / 口播稿 / Q&A / 数字卡
│   ├── 校赛答辩数据支撑.md  数字口径速查（答辩报告的精简版）
│   ├── 答辩演示脚本.md     3 分钟路演动线 + 6 个加时模块
│   ├── 接口说明.md         30 个接口的完整契约与字段含义
│   └── tables/            竞赛信息表 CSV（由 tools/export_tables.py 自动生成）
├── results/               论文真实训练产物（JSON/CSV/逐条预测）
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

完整契约（含字段含义、示例响应）见 [`docs/接口说明.md`](docs/接口说明.md)。共 31 个接口，摘要：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` · `/api/version` | 服务、设备、权重、运行环境 |
| GET | `/api/models` · `/api/checkpoints` | 模型加载详情 / 多模型对比池状态 |
| GET | `/api/overview` | 首页聚合指标（含门控与复核统计） |
| GET | `/api/experiments` · `/api/experiments/{name}` | 全部论文实验真实指标（Table 2/3/4/5） |
| GET | `/api/sqa` · `/api/seg` · `/api/datasets` | Stage1 质控 / Stage2 分割曲线 / Table1 统计 |
| GET | `/api/ablation` | Table4 消融差值表（相对完整模型） |
| GET | `/api/audit` | 数据血缘 + 隔离性 / 完整性检查（本地可复现） |
| GET | `/api/selective` · `/api/selective/{name}` | 风险-覆盖率曲线 + 校准 + 跨域分解 |
| GET | `/api/selective/experiments` · `/api/gate` | 可分析实验清单 / 当前门控与三档 |
| GET | `/api/samples` · `/api/sample_audio/{name}` | 内置样本列表 / 回放原始 wav |
| POST | `/api/analyze` · `/api/analyze_sample` | 上传或指定样本 → 完整分析 |
| POST | `/api/analyze_batch` | 批量分析（≤12 条，逐条容错，结果落库） |
| POST | `/api/compare` · `/api/compare_sample` | 同一片段多检查点并排推理 |
| POST | `/api/quality_check` | 仅六维质控（供 ESP32 硬件前置拦截） |
| GET | `/api/history` · `/api/history/{id}` | 分析历史 / 单条完整详情 |
| GET | `/api/review-queue` · `POST /api/review-queue/{id}` | 待复核队列 / 复核结论 |
| GET | `/api/stats` · `POST /api/maintenance/prune` | 统计 / 清理旧记录 |

`/api/analyze*` 均支持 `checkpoint`（选检查点）与 `gate_mode`（`loose`/`standard`/`strict`）查询参数。

---

## 6. 硬件联调约定（ESP32 数字听诊器）

* 采样格式：单声道 16-bit PCM `.wav`，建议 **4,000 Hz**（后端自动抗混叠重采样到 2 kHz）
* 时长建议 5–15 秒
* 上传：`multipart/form-data` → 先调 `/api/quality_check` 拦截低质量录音，再调 `/api/analyze`

---

## 7. 验证结果（真实样本 + 真实指标，不造数）

### 7.1 主模型在跨医院测试集上的真实表现（Table 3）

`cross_proposed_full`，训练于 PhysioNet2016 + CirCor + BMD-HS + Fraiwan，
**零样本**测试浙江大学 ZCHSound 2024（1,259 条，该库从未参与训练）：

| 实验 | MAcc/UAR | AUROC | Se | Sp | Macro-F1 | 参数 |
|---|---|---|---|---|---|---|
| 仅 PhysioNet → ZCH | 0.539 | 0.551 | 0.367 | 0.711 | 0.533 | 0.2 M |
| 多源普通混训 CNN | 0.614 | 0.667 | 0.371 | 0.857 | 0.602 | 0.2 M |
| CRNN-CBAM | 0.612 | 0.642 | 0.495 | 0.729 | 0.611 | 0.6 M |
| **Proposed Full（本系统）** | **0.664** | **0.720** | 0.539 | 0.789 | **0.664** | 22.9 M |

相对多源普通混训：**MAcc +0.050**、AUROC +0.053。

### 7.2 内置 6 条演示样本的真实推理结果

拒识门控**不是手写常数**，阈值由 `backend/selective.py` 在 7.1 的 1,259 条外测样本上反解
（标准档：不确定度 ≥ 0.3467 或决策边际 |p−0.5| < 0.10 即拒识）。MC-Dropout 已固定随机种子，
同一条音频任意次调用结果完全一致。

| 样本 | 临床真值 | 异常概率 | 不确定度 | 命中门 | 系统输出 |
|---|---|---|---|---|---|
| CirCor 13918_AV | Murmur / Abnormal | 0.401 | 0.345 | 决策边际 | **不确定 · 建议人工复核** |
| CirCor 13918_TV | Murmur / Abnormal | 0.707 | 0.315 | — | **疑似异常** ✅ |
| CirCor 13918_MV | Murmur / Abnormal | 0.047 | 0.100 | — | 正常 ❌ |
| CirCor 13918_PV | Murmur / Abnormal | 0.054 | 0.111 | — | 正常 ❌ |
| ZCHSound ZCH0001 | NORMAL | 0.158 | 0.230 | — | 正常 ✅ |
| ZCHSound ZCH0002 | NORMAL | 0.219 | 0.273 | — | 正常 ✅ |

**如实呈现局限**，不挑样本：

* 6 条中 **3 条判对、1 条转人工复核、2 条自信地判错**；
* 那 2 条判错的概率只有 0.05、不确定度仅 0.10 —— 即使用拒识也拦不住，
  说明**认知不确定度无法覆盖所有错误**，这是本方法的真实边界，不该被演示话术掩盖；
* 整体能力请以 7.1 的测试集指标为准（MAcc 0.664），不要用 6 条个例推断。

### 7.3 分割模型

6 条样本全部识别出规律 S1/S2 事件，心音周期 560–740 ms（≈ 81–107 BPM），生理范围合理；
CirCor 3,163 条上事件级 F1 **0.738**、逐帧准确率 **0.876**（见 `#research` 页训练曲线）。

### 7.4 风险-覆盖率：本端实现与论文口径逐位一致

用 `results/predictions/*.csv` 的逐条概率重新计算选择性预测，与论文
`metrics.json::selective_abstention` 对比：

| 拒识比例 | 本端重算 MAcc | 论文记录 | 覆盖率 | Se | Sp |
|---|---|---|---|---|---|
| 0% | 0.6641 | 0.6641 | 1.000 | 0.539 | 0.789 |
| 10% | **0.6775** | **0.6775** | 0.900 | 0.524 | 0.832 |
| 20% | **0.7050** | **0.7050** | 0.800 | 0.544 | 0.867 |
| 30% | 0.7231 | — | 0.700 | 0.565 | 0.881 |

点位完全重合，说明比赛端的拒识逻辑与论文实现同口径，不是另写一套近似。

三个可切换档位（阈值均在真实分布上反解）：

| 档位 | 不确定度阈值 | 边际阈值 | 实测覆盖率 | 实测 MAcc | ΔMAcc |
|---|---|---|---|---|---|
| 宽松 | 0.3511 | 0.05 | 0.854 | 0.6904 | +0.026 |
| 标准 | 0.3467 | 0.10 | 0.770 | 0.7119 | +0.048 |
| 严格 | 0.3344 | 0.15 | 0.682 | 0.7242 | +0.060 |

> 注：不确定性分布在顶部被显著压缩（p50 = 0.278、p90 = 0.351、max = 0.386），
> 因此可用阈值区间很窄。早期版本把阈值凭经验设成 0.30，实际会拒识 **44.3%** 的样本，
> 远高于设计意图；改为数据反解后才落到 10–30% 的目标区间。

### 7.5 消融实验：五个机制全部是正贡献

`Δ = 完整模型 − 消融项`，正值表示移除后性能下降（即该机制有贡献）。**5/5 全部完成**：

| 移除的机制 | MAcc | ΔMAcc | ΔAUROC | Se | Sp |
|---|---|---|---|---|---|
| CWT 双视图 | 0.6005 | **+0.0636** | +0.0757 | 0.475 | 0.726 |
| MixStyle 风格随机化 | 0.6247 | +0.0394 | +0.0449 | 0.447 | 0.802 |
| 肺音/频响增强 | 0.6429 | +0.0212 | +0.0332 | 0.449 | 0.837 |
| CBAM 注意力 | 0.6471 | +0.0170 | +0.0381 | 0.472 | 0.823 |
| GRL 域对抗 | 0.6509 | +0.0132 | +0.0380 | 0.528 | 0.773 |

五项 ΔMAcc 与 ΔAUROC **全部为正**，贡献排序 CWT > MixStyle > 增强 > CBAM > GRL。
其中 CWT 双视图的贡献（+0.064）超过其后三项之和，说明**时频双视图是本方案的主承重结构**，
而 CBAM / GRL 属于边际增益（+0.017 / +0.013），在论文里不应表述为“显著提升”。

### 7.6 跨设备域分解：噪声采集是真正的短板

ZCHSound 外测集内部本身含两个采集域：

| 域 | 占比 | 不给结论时 MAcc | Se | Sp | 开拒识后 MAcc | 覆盖率 |
|---|---|---|---|---|---|---|
| 清洁采集 `zch_clean` | 74.7% | **0.7189** | 0.635 | 0.803 | 0.7695 | 78.6% |
| 噪声采集 `zch_noisy` | 25.3% | **0.5174** | 0.291 | 0.744 | 0.5511 | 72.0% |

**域间差距 0.2015**，且塌方主要来自敏感度（0.635 → 0.291，漏诊翻倍）。
这比“总体 MAcc 0.664”更有信息量：模型在同一家医院、换一种采集条件就接近失效，
说明数据增强（`abl_no_aug`）与采集质控（六维 SQA）不是可选项而是必需项。

### 7.7 概率校准：两类都偏过自信约 9 个百分点

可靠度图（`#research` 页可交互查看）显示：

* 期望校准误差 **ECE = 0.0902**（论文记录 0.0893，口径基本一致）；
* 被判为异常的一批：平均置信度 0.767，实际患病率 0.676；
* 被判为正常的一批：平均置信度 0.766，实际正常率 0.677。

两类**同向过自信**，幅度接近，因此排序类指标（AUROC）不受影响，但**概率数值不能直接当临床风险读数用**。
这也是本系统坚持「概率 + 不确定度 + 拒识」三件套、而不是只报一个概率的原因。

> ⚠️ 关于 `lodo_proposed_dg` 的说明：它在上述 6 条演示样本上可全部判对，
> 但它并非论文主模型（Table 2 中其 LODO MAcc 0.626 与基线 0.627 基本持平，是**中性结果**）。
> 我们仍默认加载论文主模型 `cross_proposed_full`，把所有错误一并展示；
> LODO 模型作为「多模型对比」页的一个可切换检查点存在，用于说明域泛化机制的差异，
> 不作为“换个权重就能全对”的演示捷径。

### 7.8 Table 5：单库内部 5 折（不是"跑了一折就叫 5 折"）

HFrEF 2025 共 2,480 条 / 620 人，其中 fold 0–4 各 400 条（各 100 人），另 480 条（120 人）
`fold_internal = -2` 始终只参与训练、不进任何测试折。**5 折全部跑完**后聚合：

| 指标 | 5 折均值 ± 标准差 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| MAcc/UAR | **0.7052 ± 0.0279** | 0.7296 | 0.7282 | 0.6686 | 0.7169 | 0.6828 |
| AUROC | **0.7634 ± 0.0422** | 0.8120 | 0.7989 | 0.7195 | 0.7237 | 0.7631 |
| Se | 0.6292 ± 0.0948 | | | | | |
| Sp | 0.7813 ± 0.1014 | | | | | |
| Macro-F1 | 0.6245 ± 0.0571 | | | | | |

> ⚠️ 这一项原先只跑了 fold 0 却命名为 `internal_hfref_5fold`，而 fold 0 恰好是
> 5 折里**最高**的一折（0.7296）。若照原样报出，等于择优单折冒充 5 折。
> 补跑 fold 1–4 后真实均值是 **0.7052**，低了 0.024 —— 差一个标准差还多。
> 修法与聚合脚本见 论文侧的「修复记录」文档（随论文保存，不在本仓库） 第 5 条。

### 7.9 论文图表产物状态

| 图 | 内容 | 来源 | 状态 |
|---|---|---|---|
| Fig.2 | LODO 嵌入 t-SNE（捷径 vs 域泛化） | `visualize_paper.py --mode tsne` | Stage 5 自动 |
| Fig.3 | 风险-覆盖率曲线 | `--mode risk_coverage` | Stage 5 自动 |
| Fig.4 | 白盒案例：波形 + Log-Mel + Grad-CAM | `--mode gradcam` | **已生成 3 例**（原流水线漏掉此步） |
| Fig.5 | 混淆矩阵 + ROC | `--mode cm_roc` | Stage 5 自动 |

Fig.4 按论文交接文档要求「1 条正常 + 2 条典型杂音」生成了 3 个病例：

| 文件 | 病例 | 真值 | 模型判定 |
|---|---|---|---|
| `fig4_gradcam_case1.png` | ZCHSound `ZCH0001` | 正常 | 正常（p=0.158） |
| `fig4_gradcam_case2.png` | CirCor `13918_TV` | 杂音 | 异常（p=0.707）✅ |
| `fig4_gradcam_case3.png` | CirCor `13918_AV` | 杂音 | 不确定 · 拒识（p=0.401） |

case3 是有意保留的「模型自己也不确定」病例，正文可直接用于讨论拒识机制。

生成过程中修掉论文脚本四处缺陷（`np.trapz` 移除、Grad-CAM 钩子返回值破坏 autograd、
裁剪口径与评估不一致、Fig.4 未进流水线），详见论文侧的「修复记录」文档（随论文保存，不在本仓库）。
修正后该图概率由 0.911 变为 **0.701**，与论文表格及本演示端同一病例的 0.707 一致
（差异仅来自 MC-Dropout 采样 vs 单次确定性前向）。

---

## 8. 数据与合规声明

四类内容**不纳入版本库**，一律由 `tools/sync_results.py` 从训练服务器获取：

| 内容 | 原因 |
|---|---|
| `samples/` | 原始临床录音（CirCor / ZCHSound） |
| `weights/` | 由受控数据派生的训练权重（约 230 MB） |
| `results/predictions/` | **行级**预测：含记录编号 + 临床标签 + 逐条概率 |
| `results/master_pcg_manifest.csv` | **行级**临床清单：含受试者 ID、医院内部路径、年龄段、站点、临床分级 |

CirCor DigiScope 与 PhysioNet/CinC 2016 属**受控访问（credentialed access）**数据集，
ZCHSound 为医院内部数据，公开发布原始录音、由其派生的权重、或行级标注都会违反数据使用协议。

### 8.1 清单为什么放脱敏版

仓库里入库的是 `results/master_pcg_manifest.public.csv`（由 `tools/sanitize_manifest.py` 生成）：

| 原始列 | 脱敏后 | 说明 |
|---|---|---|
| `subject_id` | `subject_hash` | `sha256("sonotrace:数据集\|受试者")[:12]`，同一受试者哈希恒定 |
| `rel_path` | `path_present` (0/1) | 保留「是否关联到音频」的判定，不泄露任何真实路径 |
| `age_group` / `site` / `label_fine` / `label_binary_clinical` / `site_has_murmur` / `sqi_label` / `seg_tsv_rel_path` / `record_uid` | **移除** | 与统计与核验无关 |

**脱敏不损失可验证性**：因为哈希对同一受试者恒定，
「外测集受试者隔离」与「受试者级折内唯一」两项检查仍然成立。
实测两种清单（原始 / 脱敏）跑出的结果**完全一致**：
11,538 条 / 5,659 人 / 75.36 h / 14 域，审计 **6/6 通过**。

### 8.2 临床合规

* 本系统输出**仅为辅助分析参考，不替代专业医师诊断**。界面与报告中均含此声明。

---

## 9. 与论文目录的关系

| 目录 | 职责 | 是否可公开 |
|---|---|---|
| `paper/` | 论文训练脚本、5 折防泄漏清单、消融实验流水线 | 代码可公开；manifests 中不含音频本体 |
| `sonotrace-web/` | 比赛展示端前后端（本仓库） | 可公开（权重/样本除外） |

两者共用同一份预处理口径与模型结构定义，保证"论文指标 = 演示结果"。