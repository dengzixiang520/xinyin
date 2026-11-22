# -*- coding: utf-8 -*-
"""
声见其迹 (SONOTRACE) — 比赛端统一配置

所有信号处理与模型超参 **严格对齐论文训练时的 configs/experiment_config.yaml**，
以保证比赛端推理与论文实验的数值口径完全一致（否则论文指标与演示结果会对不上）。

论文来源: paper/configs/experiment_config.yaml
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- 路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ---------------------------------------------------------------- 音频口径 (对齐论文)
AUDIO = {
    "target_sr": 2000,        # 统一重采样率
    "bandpass_low": 25.0,     # 带通下限 Hz
    "bandpass_high": 400.0,   # 带通上限 Hz
    "env_sr": 50,             # 分割 4 通道包络降采样率
    "seg_win_len": 512,       # 1D U-Net 输入帧数 (512 @ 50Hz ≈ 10.24s)
    "cls_clip_sec": 5.0,      # 分类滑窗时长
    "n_mels": 64,             # Log-Mel 频率维数
    "n_fft": 256,
    "hop_length": 64,
    "cwt_scales": 64,         # CWT 尺度数
}

# 分割模型结构 (对齐论文 train_seg.py: PCGUNet1D(in_ch=4, num_classes=4, base_ch=16))
SEG = {"in_ch": 4, "num_classes": 4, "base_ch": 16,
       "state_names": ["S1", "收缩期", "S2", "舒张期"]}

# 分类推理期 MC-Dropout 采样次数 (对齐论文 mc_dropout_passes)
MC_PASSES = 10

# 六维 SQA 特征顺序 (必须与 train_sqa.py 的 feat_cols 完全一致!)
SQA_FEATURES = ["duration_sec", "rms_energy", "silence_ratio",
                "clipping_ratio", "snr_proxy_db", "stability_cv"]

SQA_FEATURE_LABELS = {
    "duration_sec": "录音时长 (s)",
    "rms_energy": "均方根能量",
    "silence_ratio": "静音帧比例",
    "clipping_ratio": "峰值削波比例",
    "snr_proxy_db": "频带信噪比 (dB)",
    "stability_cv": "能量稳定系数 CV",
}

# ---------------------------------------------------------------- 权重文件名
CLS_WEIGHT = os.path.join(WEIGHTS_DIR, "cls_best.pt")
SEG_WEIGHT = os.path.join(WEIGHTS_DIR, "seg_unet.pt")
SQA_WEIGHT = os.path.join(WEIGHTS_DIR, "sqa_rf_model.pkl")

# ---------------------------------------------------------------- 多模型对比池
# 「同一片段跑多个检查点」用的注册表。文件缺失的项在前端自动隐藏, 不会报错。
# 每一项都对应论文里一个真实训练产物, 不是临时训练的对照模型。
COMPARE_CHECKPOINTS = {
    "proposed": {
        "path": CLS_WEIGHT,
        "label": "Proposed Full",
        "desc": "双视图 + CBAM + MixStyle + GRL（论文主模型）",
        "table": "Table3",
    },
    "naive": {
        "path": os.path.join(WEIGHTS_DIR, "cls_naive.pt"),
        "label": "普通混训 CNN",
        "desc": "同样四库混训，去掉全部域泛化机制（0.2M 参数）",
        "table": "Table3",
    },
    "lodo": {
        "path": os.path.join(WEIGHTS_DIR, "cls_lodo.pt"),
        "label": "LODO 域泛化",
        "desc": "PhysioNet 留一子库训练的域泛化版本",
        "table": "Table2",
    },
    "nocwt": {
        "path": os.path.join(WEIGHTS_DIR, "cls_nocwt.pt"),
        "label": "去 CWT 双视图",
        "desc": "消融：仅 Log-Mel 单视图",
        "table": "Table4",
    },
}
DEFAULT_CHECKPOINT = "proposed"

# ---------------------------------------------------------------- 拒识门控
# 阈值不写死: 运行时由 selective.py 在论文真实外测集上反解 (见 backend/selective.py)。
# 这里的值仅在标定数据缺失时作兜底。
ABSTAIN_DEFAULT = {"uncertainty_threshold": 0.35, "margin_threshold": 0.10}

APP_TITLE = "声见其迹 SONOTRACE"
APP_VERSION = "3.1"


def checkpoint_status() -> dict:
    """对比池里每个检查点的落盘状态。"""
    out = {}
    for key, spec in COMPARE_CHECKPOINTS.items():
        p = spec["path"]
        ok = os.path.exists(p)
        out[key] = {"label": spec["label"], "desc": spec["desc"],
                    "table": spec["table"], "file": os.path.basename(p),
                    "available": ok,
                    "mb": round(os.path.getsize(p) / 1048576, 2) if ok else 0.0}
    return out


def weights_status() -> dict:
    """返回三项权重的落盘状态与文件大小 (MB)。"""
    out = {}
    for key, path in (("cls", CLS_WEIGHT), ("seg", SEG_WEIGHT), ("sqa", SQA_WEIGHT)):
        if os.path.exists(path):
            out[key] = {"loaded": True, "mb": round(os.path.getsize(path) / 1048576, 2),
                        "file": os.path.basename(path)}
        else:
            out[key] = {"loaded": False, "mb": 0.0, "file": os.path.basename(path)}
    return out