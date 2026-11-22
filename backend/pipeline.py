# -*- coding: utf-8 -*-
"""
声见其迹 — 单样本端到端分析流水线

一次调用返回比赛前端四个模块需要的全部数据:
  1) 六维前置质控 (duration/rms/silence/clipping/snr/stability + RF 概率 + 问题清单)
  2) 声像对照  (波形 500 点 + Log-Mel 谱 + 4 通道包络 + S1/S2 事件时间戳)
  3) 智能分析  (异常概率 + MC-Dropout 不确定性 + Grad-CAM + 256 维嵌入)
  4) 报告文案  (结构化结论 + 临床合规声明)

判定采用论文的风险-覆盖率机制, 两道门控参数 **不写死**, 由
backend/selective.py 在 cross_proposed_full 的零样本外测集 (n=1259) 上反解:
    门 1  认知不确定度  uncertainty >= thr   (标准档 thr≈0.3467, 拒识 20%)
    门 2  决策边际      |prob-0.5| < margin (标准档 0.10)
命中任一即输出「建议人工复核」, 不给自动结论。
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import config as C
from . import selective as SEL
from .engine import (load_audio_from_path, extract_quality, predict_cls,
                     predict_seg, compute_gradcam, extract_dualview, REGISTRY)

# 门控参数的标定来源实验 (论文零样本外测集, 覆盖 zch_clean + zch_noisy 两个域)
GATE_EXPERIMENT = "cross_proposed_full"


def _resample_points(y: np.ndarray, n: int = 500) -> list:
    if len(y) == 0:
        return [0.0] * n
    idx = np.linspace(0, len(y) - 1, n)
    return np.interp(idx, np.arange(len(y)), y).round(4).tolist()


def _clip(y: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    n = int(sr * seconds)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def current_gate(mode: Optional[str] = None) -> dict:
    """取当前生效的门控参数 (含标定来源, 供前端展示阈值出处)。"""
    if mode:
        g = SEL.gate_for_mode(GATE_EXPERIMENT, mode)
    else:
        g = SEL.default_gate(GATE_EXPERIMENT)
    if g.get("uncertainty_threshold") is None:
        g["uncertainty_threshold"] = C.ABSTAIN_DEFAULT["uncertainty_threshold"]
        g["calibrated"] = False
    if g.get("margin_threshold") is None:
        g["margin_threshold"] = C.ABSTAIN_DEFAULT["margin_threshold"]
    return g


def _decide(quality: dict, cls: dict, gate: dict):
    """双门控判定。返回 (verdict, verdict_key, reason, gate_hit)。"""
    thr = gate.get("uncertainty_threshold")
    mg = gate.get("margin_threshold") or 0.0

    if not quality["passed"]:
        return ("采集质量未达标 · 待复核", "pending",
                "六维质控未通过，建议重新采集后再判定", "quality")

    if not cls.get("ready"):
        return ("声学特征分析 (模型待载入)", "pending",
                "尚未放入分类权重，当前仅输出 DSP 声学特征分析", "no_model")

    p = cls.get("prob_abnormal")
    unc = cls.get("uncertainty")

    # 门 1: 认知不确定度
    if thr is not None and unc is not None and unc >= thr:
        return ("不确定 · 建议人工复核", "pending",
                f"MC-Dropout 认知不确定度 {unc:.4f} ≥ 门控阈值 {thr:.4f}"
                f"（异常概率仅 {(p or 0) * 100:.1f}%），"
                f"按选择性拒识策略不作自动结论", "uncertainty")

    # 门 2: 决策边际
    if p is not None and abs(p - 0.5) < mg:
        return ("不确定 · 建议人工复核", "pending",
                f"异常概率 {p * 100:.1f}% 距决策边界不足 {mg * 100:.0f} 个百分点"
                f"（边际 {abs(p - 0.5):.4f} < {mg}），等价于掷硬币，"
                f"按选择性拒识策略不作自动结论", "margin")

    if p >= 0.5:
        return ("疑似异常心音", "abnormal",
                f"模型判定异常概率 {p * 100:.1f}%，"
                f"且不确定度 {unc:.4f} 低于门控阈值 {thr:.4f}", "none")
    return ("正常生理心音", "normal",
            f"模型判定正常概率 {(1 - p) * 100:.1f}%，"
            f"不确定度 {unc:.4f} 低于门控阈值 {thr:.4f}", "none")


def analyze(wav_path: str, filename: str = "", include_gradcam: bool = True,
            checkpoint: Optional[str] = None, gate_mode: Optional[str] = None,
            with_embedding: bool = False) -> dict:
    ckpt = checkpoint or C.DEFAULT_CHECKPOINT
    y, raw_y, sr = load_audio_from_path(wav_path)
    dur = len(raw_y) / float(sr)

    quality = extract_quality(raw_y, sr)

    # ---------------- 声像对照 ----------------
    vis_sec = min(5.0, dur)
    y_vis = _clip(y, max(vis_sec, 0.1), sr)

    seg = predict_seg(y)          # 分割用完整音频 (支持最长 10.24s 窗)

    # ---------------- 智能分析 ----------------
    # 双视图特征属纯 DSP, 与权重无关 —— 声像对照模块始终可用
    _, dsp_mel, dsp_cwt = extract_dualview(y, checkpoint=ckpt)
    cls = predict_cls(y, checkpoint=ckpt)
    cam = compute_gradcam(y, checkpoint=ckpt) if (include_gradcam and cls.get("ready")) else None

    gate = current_gate(gate_mode)
    verdict, verdict_key, reason, gate_hit = _decide(quality, cls, gate)

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
        "checkpoint": ckpt,
        "checkpoint_label": REGISTRY.cls_meta(ckpt).get("label", ckpt),
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
            "gate_hit": gate_hit,
            "abstain_threshold": gate.get("uncertainty_threshold"),
            "margin_threshold": gate.get("margin_threshold"),
            "gate_mode": gate.get("mode"),
            "gate_calibrated": gate.get("calibrated"),
            "gate_calibrated_on": gate.get("calibrated_on"),
            "abstained": verdict_key == "pending" and quality["passed"]
                         and cls.get("ready"),
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
        **({"embedding": cls.get("embedding")} if with_embedding else {}),
    }


def analyze_sample(name: str, include_gradcam: bool = True,
                   checkpoint: Optional[str] = None,
                   gate_mode: Optional[str] = None) -> dict:
    path = os.path.join(C.SAMPLES_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    return analyze(path, filename=name, include_gradcam=include_gradcam,
                   checkpoint=checkpoint, gate_mode=gate_mode)


def compare(wav_path: str, filename: str = "",
            checkpoints: Optional[list] = None) -> dict:
    """同一片段在多检查点上并排推理 —— 域泛化机制到底带来了什么, 肉眼可见。

    只返回各模型的概率/不确定度/判定, 不返回谱图与 Grad-CAM (省带宽)。
    """
    pool = C.COMPARE_CHECKPOINTS
    want = [k for k in (checkpoints or list(pool.keys())) if k in pool]
    y, raw_y, sr = load_audio_from_path(wav_path)
    gate = current_gate(None)
    quality = extract_quality(raw_y, sr)

    rows = []
    for key in want:
        cls = predict_cls(y, checkpoint=key)
        verdict, vkey, reason, hit = _decide(quality, cls, gate)
        meta = REGISTRY.cls_meta(key)
        rows.append({
            "checkpoint": key,
            "label": pool[key]["label"],
            "desc": pool[key]["desc"],
            "table": pool[key]["table"],
            "ready": cls.get("ready", False),
            "error": cls.get("error"),
            "prob_abnormal": cls.get("prob_abnormal"),
            "prob_normal": cls.get("prob_normal"),
            "uncertainty": cls.get("uncertainty"),
            "mc_samples": cls.get("mc_samples"),
            "verdict": verdict,
            "verdict_key": vkey,
            "gate_hit": hit,
            "params_M": meta.get("params_M"),
            "use_cwt": meta.get("use_cwt"),
            "use_cbam": meta.get("use_cbam"),
            "use_mixstyle": meta.get("use_mixstyle"),
            "use_grl": meta.get("use_grl"),
        })

    agree = None
    keys = {r["verdict_key"] for r in rows if r["ready"]}
    if len(keys) == 1:
        agree = True
    elif len(keys) > 1:
        agree = False
    return {
        "filename": filename,
        "duration_sec": round(len(raw_y) / float(sr), 2),
        "quality_passed": quality["passed"],
        "gate": gate,
        "models": rows,
        "all_agree": agree,
        "note": ("各模型对同一段录音的结论不完全一致时, 差异本身即为域泛化机制"
                 "有效性的直接证据; 一致时说明该片段对模型结构不敏感。"),
    }


def list_samples() -> list:
    out = []
    if not os.path.isdir(C.SAMPLES_DIR):
        return out
    for n in sorted(os.listdir(C.SAMPLES_DIR)):
        if n.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
            p = os.path.join(C.SAMPLES_DIR, n)
            out.append({"name": n, "kb": round(os.path.getsize(p) / 1024, 1)})
    return out