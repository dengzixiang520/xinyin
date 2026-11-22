# -*- coding: utf-8 -*-
"""
声见其迹 — 单样本端到端分析流水线

一次调用返回比赛前端四个模块需要的全部数据:
  1) 六维前置质控 (duration/rms/silence/clipping/snr/stability + RF 概率 + 问题清单)
  2) 声像对照  (波形 500 点 + Log-Mel 谱 + 4 通道包络 + S1/S2 事件时间戳)
  3) 智能分析  (异常概率 + MC-Dropout 不确定性 + Grad-CAM + 256 维嵌入)
  4) 报告文案  (结构化结论 + 临床合规声明)
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .engine import (load_audio_from_path, extract_quality, predict_cls,
                     predict_seg, compute_gradcam, extract_dualview, REGISTRY)


def _resample_points(y: np.ndarray, n: int = 500) -> list:
    if len(y) == 0:
        return [0.0] * n
    idx = np.linspace(0, len(y) - 1, n)
    return np.interp(idx, np.arange(len(y)), y).round(4).tolist()


def _clip(y: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    n = int(sr * seconds)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def analyze(wav_path: str, filename: str = "", include_gradcam: bool = True) -> dict:
    y, raw_y, sr = load_audio_from_path(wav_path)
    dur = len(raw_y) / float(sr)

    quality = extract_quality(raw_y, sr)

    # ---------------- 声像对照 ----------------
    vis_sec = min(5.0, dur)
    y_vis = _clip(y, max(vis_sec, 0.1), sr)

    seg = predict_seg(y)          # 分割用完整音频 (支持最长 10.24s 窗)

    # ---------------- 智能分析 ----------------
    # 双视图特征属纯 DSP, 与权重无关 —— 声像对照模块始终可用
    _, dsp_mel, dsp_cwt = extract_dualview(y)
    cls = predict_cls(y)
    cam = compute_gradcam(y) if (include_gradcam and cls.get("ready")) else None

    # ---------------- 结论与文案 ----------------
    if not quality["passed"]:
        verdict = "不确定 / 待复核"
        verdict_key = "pending"
        reason = "采集质量未达标，建议重新采集后再判定"
    elif cls.get("ready"):
        if cls["prob_abnormal"] >= 0.5:
            verdict = "疑似异常心音"
            verdict_key = "abnormal"
            reason = f"模型判定异常概率 {cls['prob_abnormal']*100:.1f}%，且 MC-Dropout 不确定度 {cls['uncertainty']:.3f}"
        else:
            verdict = "正常生理心音"
            verdict_key = "normal"
            reason = f"模型判定正常概率 {cls['prob_normal']*100:.1f}%，不确定度 {cls['uncertainty']:.3f}"
    else:
        # 无分类权重时的物理兜底: 中高频能量占比
        verdict = "声学特征分析 (模型待载入)"
        verdict_key = "pending"
        reason = "尚未放入 cls_best.pt，当前仅输出 DSP 声学特征分析"

    f = quality["features"]
    explanation = (
        f"录音有效时长 {f['duration_sec']}s，均方根能量 {f['rms_energy']:.4f}，"
        f"频带信噪比估计 {f['snr_proxy_db']} dB，静音帧比例 {f['silence_ratio']:.3f}，"
        f"能量稳定系数 {f['stability_cv']:.3f}。"
        + (f"分割模型识别到 S1 事件 {len(seg.get('s1_centers_sec', []))} 个、"
           f"S2 事件 {len(seg.get('s2_centers_sec', []))} 个。" if seg.get("ready") else "")
        + f" 结论：{reason}。本系统提供辅助分析参考，不替代专业医师诊断。"
    )

    s1 = seg.get("s1_centers_sec", []) if seg.get("ready") else []
    s2 = seg.get("s2_centers_sec", []) if seg.get("ready") else []
    cycle_ms = None
    if len(s1) >= 2:
        cycle_ms = round(float(np.median(np.diff(s1))) * 1000, 1)

    return {
        "filename": filename,
        "engine_mode": "real_pytorch_model" if cls.get("ready") else "dsp_acoustic_fallback",
        "weights_loaded": C.weights_status(),
        "duration_sec": round(dur, 2),
        "sample_rate": sr,
        "quality": quality,
        "visualization": {
            "vis_duration_sec": round(vis_sec, 2),
            "waveform_500pts": _resample_points(y_vis, 500),
            "mel": dsp_mel.round(4).tolist(),
            "cwt": dsp_cwt.round(4).tolist(),
            "mel_shape": list(dsp_mel.shape),
            "mel_freq_range_hz": [20.0, 800.0],
            "time_axis_sec": round(len(dsp_mel[0]) * C.AUDIO["hop_length"] / C.AUDIO["target_sr"], 3),
        },
        "segmentation": {
            "ready": seg.get("ready", False),
            "error": seg.get("error"),
            "state_names": C.SEG["state_names"],
            "env_sr": seg.get("env_sr"),
            "frames": seg.get("frames"),
            "state_seq": seg.get("state_seq"),
            "s1_centers_sec": s1,
            "s2_centers_sec": s2,
            "cycle_ms": cycle_ms,
            "envelopes_4ch": seg.get("envelopes_4ch"),
            "status": seg.get("status"),
        },
        "prediction": {
            "ready": cls.get("ready", False),
            "error": cls.get("error"),
            "label": verdict,
            "verdict_key": verdict_key,
            "prob_abnormal": cls.get("prob_abnormal"),
            "prob_normal": cls.get("prob_normal"),
            "uncertainty": cls.get("uncertainty"),
            "mc_passes": cls.get("mc_passes"),
            "mc_samples": cls.get("mc_samples"),
            "explanation": explanation,
            "meta": cls.get("meta"),
            "embedding_dim": len(cls["embedding"]) if cls.get("embedding") else None,
        },
        "gradcam": cam,
    }


def analyze_sample(name: str, include_gradcam: bool = True) -> dict:
    import os
    path = os.path.join(C.SAMPLES_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    return analyze(path, filename=name, include_gradcam=include_gradcam)


def list_samples() -> list:
    import os
    out = []
    if not os.path.isdir(C.SAMPLES_DIR):
        return out
    for n in sorted(os.listdir(C.SAMPLES_DIR)):
        if n.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
            p = os.path.join(C.SAMPLES_DIR, n)
            out.append({"name": n, "kb": round(os.path.getsize(p) / 1024, 1)})
    return out