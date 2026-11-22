# -*- coding: utf-8 -*-
"""
声见其迹 — 真实模型推理引擎

三项权重的加载与推理:
  cls_best.pt      DualViewDGResNetCBAM (Log-Mel + CWT 双视图 + CBAM + MixStyle + GRL)
                   -> 二分类异常概率 + MC-Dropout 不确定性 + Grad-CAM 热力图 + 256 维嵌入
  seg_unet.pt      PCGUNet1D 四状态分割
                   -> 逐帧四状态概率 + 生理 Viterbi 解码后的 S1/S2 事件时间戳
  sqa_rf_model.pkl 六维随机森林质控
                   -> 采集质量合格概率

所有推理参数严格对齐论文训练口径 (见 config.py)。
权重缺失时优雅降级为纯 DSP 声学待命模式，绝不抛错 —— 保证答辩现场零翻车。
"""
from __future__ import annotations

import os
import pickle
import threading
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# sklearn 版本与训练环境不同(1.7.2 vs 1.8.x)时只会给出不影响推理的警告, 这里静音
warnings.filterwarnings("ignore", category=UserWarning)
try:  # scikit-learn >= 1.7
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:                                                # noqa: BLE001
    pass

from . import config as C
from .preprocess import (
    load_and_preprocess_audio,
    compute_4ch_envelopes,
    extract_log_mel,
    extract_cwt_scalogram,
    extract_sqa_6d_features,
)
from .models.cls_models import build_model, MixStyle, CBAM
from .models.seg_unet import PCGUNet1D, viterbi_cardiac_decode, extract_event_centers

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOCK = threading.Lock()

# MC-Dropout 采样种子: 固定以保证「同一条录音 → 同一个结论」可复现
_MC_SEED = 20241122


def _safe_torch_load(path: str, map_location="cpu"):
    """兼容 torch>=2.6 的 weights_only 默认翻转 (论文 ckpt 存了 argparse 字典)。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class ModelRegistry:
    """分割/SQA 单例 + 分类检查点池 (多模型对比按需惰性加载)。"""

    def __init__(self):
        self._cls_pool: dict[str, dict] = {}      # name -> {tried, model, meta, error}
        self.seg_model = None
        self.seg_error: Optional[str] = None
        self.sqa_bundle = None
        self.sqa_error: Optional[str] = None
        self._tried = {"seg": False, "sqa": False}

    # ------------------------------------------------------------ 分类检查点池
    def _build_cls(self, name: str) -> dict:
        """按 checkpoint 自带的 args 重建网络结构 (消融变体也能正确加载)。"""
        spec = C.COMPARE_CHECKPOINTS.get(name)
        if spec is None:
            return {"tried": True, "model": None, "meta": {},
                    "error": f"未注册的检查点: {name}"}
        path = spec["path"]
        if not os.path.exists(path):
            return {"tried": True, "model": None, "meta": {},
                    "error": f"未找到 {os.path.basename(path)}"}
        try:
            ckpt = _safe_torch_load(path, map_location="cpu")
            a = ckpt.get("args", {}) or {}
            model_name = a.get("model", "dg_resnet_cbam")
            domain_to_idx = ckpt.get("domain_to_idx", {}) or {}
            use_cwt = (not a.get("no_cwt", False)) and (model_name == "dg_resnet_cbam")
            kwargs = {}
            if model_name == "dg_resnet_cbam":
                kwargs = dict(use_cwt=use_cwt,
                              use_cbam=not a.get("no_cbam", False),
                              use_mixstyle=not a.get("no_mixstyle", False),
                              use_grl=not a.get("no_grl", False))
            model = build_model(model_name, num_classes=2,
                                num_domains=max(1, len(domain_to_idx)), **kwargs)
            model.load_state_dict(ckpt["model_state"], strict=False)
            model.to(_DEVICE).eval()
            meta = {
                "model": model_name,
                "checkpoint": name,
                "label": spec["label"],
                "use_cwt": bool(use_cwt),
                "use_cbam": bool(kwargs.get("use_cbam", False)),
                "use_mixstyle": bool(kwargs.get("use_mixstyle", False)),
                "use_grl": bool(kwargs.get("use_grl", False)),
                "exp_name": a.get("exp_name", "unknown"),
                "protocol": a.get("protocol", "unknown"),
                "num_domains": len(domain_to_idx),
                "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                "device": str(_DEVICE),
            }
            return {"tried": True, "model": model, "meta": meta, "error": None}
        except Exception as ex:                                   # noqa: BLE001
            return {"tried": True, "model": None, "meta": {},
                    "error": f"{type(ex).__name__}: {ex}"}

    def ensure_cls(self, name: Optional[str] = None):
        name = name or C.DEFAULT_CHECKPOINT
        ent = self._cls_pool.get(name)
        if ent is None:
            ent = self._build_cls(name)
            self._cls_pool[name] = ent
            if ent["error"]:
                warnings.warn(f"[cls:{name}] {ent['error']}")
        return ent["model"]

    def cls_meta(self, name: Optional[str] = None) -> dict:
        name = name or C.DEFAULT_CHECKPOINT
        self.ensure_cls(name)
        return self._cls_pool.get(name, {}).get("meta", {})

    def cls_error(self, name: Optional[str] = None) -> Optional[str]:
        name = name or C.DEFAULT_CHECKPOINT
        self.ensure_cls(name)
        return self._cls_pool.get(name, {}).get("error")

    # 兼容旧调用点 (REGISTRY.cls_model / REGISTRY.cls_meta / REGISTRY.cls_error)
    @property
    def cls_model(self):
        return self.ensure_cls()

    def pool_report(self, load_all: bool = False) -> dict:
        """检查点池状态。默认只真正加载主模型, 其余按文件存在性报告,
        避免开服时把 4 个权重(ca. 300MB) 全塞进内存。"""
        out = {}
        for key, spec in C.COMPARE_CHECKPOINTS.items():
            ent = self._cls_pool.get(key) or {}
            exists = os.path.exists(spec["path"])
            loaded = ent.get("model") is not None
            if load_all and exists and not ent.get("tried"):
                self.ensure_cls(key)
                ent = self._cls_pool.get(key) or {}
                loaded = ent.get("model") is not None
            out[key] = {
                "label": spec["label"], "desc": spec["desc"], "table": spec["table"],
                "file": os.path.basename(spec["path"]),
                "available": exists,
                "loaded": loaded,
                "ready": exists,                  # 可用即可被「模型对比」调用
                "error": ent.get("error"),
                **{k: v for k, v in (ent.get("meta") or {}).items()
                   if k in ("model", "params_M", "exp_name", "protocol",
                            "use_cwt", "use_cbam", "use_mixstyle", "use_grl",
                            "num_domains")},
            }
        return out

    # ------------------------------------------------------------ 分割
    def ensure_seg(self):
        if self._tried["seg"]:
            return self.seg_model
        self._tried["seg"] = True
        if not os.path.exists(C.SEG_WEIGHT):
            self.seg_error = "未找到 seg_unet.pt"
            return None
        try:
            sd = _safe_torch_load(C.SEG_WEIGHT, map_location="cpu")
            if isinstance(sd, dict) and "model_state" in sd:
                sd = sd["model_state"]
            # 只传结构参数 (C.SEG 里还有 state_names 等展示用键, 不能传给 __init__)
            seg_kwargs = {k: C.SEG[k] for k in ("in_ch", "num_classes", "base_ch")}
            model = PCGUNet1D(**seg_kwargs)
            model.load_state_dict(sd, strict=False)
            model.to(_DEVICE).eval()
            self.seg_model = model
            return model
        except Exception as ex:                                   # noqa: BLE001
            self.seg_error = f"{type(ex).__name__}: {ex}"
            self.seg_model = None
            return None

    # ------------------------------------------------------------ SQA
    def ensure_sqa(self):
        if self._tried["sqa"]:
            return self.sqa_bundle
        self._tried["sqa"] = True
        if not os.path.exists(C.SQA_WEIGHT):
            self.sqa_error = "未找到 sqa_rf_model.pkl"
            return None
        try:
            with open(C.SQA_WEIGHT, "rb") as f:
                self.sqa_bundle = pickle.load(f)
            return self.sqa_bundle
        except Exception as ex:                                   # noqa: BLE001
            self.sqa_error = f"{type(ex).__name__}: {ex}"
            self.sqa_bundle = None
            return None

    # ------------------------------------------------------------ 状态
    def status(self) -> dict:
        ws = C.weights_status()
        pool = self.pool_report(load_all=False)
        self.ensure_cls()                  # 主模型真正加载一次, 保证 ready 是真值
        self.ensure_seg(); self.ensure_sqa()
        pool[C.DEFAULT_CHECKPOINT]["loaded"] = \
            self._cls_pool.get(C.DEFAULT_CHECKPOINT, {}).get("model") is not None
        pool[C.DEFAULT_CHECKPOINT]["ready"] = pool[C.DEFAULT_CHECKPOINT]["loaded"]
        if self._cls_pool.get(C.DEFAULT_CHECKPOINT, {}).get("meta"):
            pool[C.DEFAULT_CHECKPOINT].update(
                {k: v for k, v in self._cls_pool[C.DEFAULT_CHECKPOINT]["meta"].items()
                 if k in ("model", "params_M", "exp_name", "protocol", "use_cwt",
                          "use_cbam", "use_mixstyle", "use_grl", "num_domains")})
        default = pool.get(C.DEFAULT_CHECKPOINT, {})
        engine_ok = bool(default.get("ready"))
        return {
            "weights": ws,
            "checkpoints": pool,
            "default_checkpoint": C.DEFAULT_CHECKPOINT,
            "n_available": sum(1 for v in pool.values() if v.get("available")),
            # 兼容旧前端字段
            "cls": {"ready": engine_ok, "error": default.get("error"),
                    **{k: v for k, v in default.items()
                       if k in ("model", "params_M", "exp_name", "protocol",
                                "use_cwt", "use_cbam", "use_mixstyle", "use_grl",
                                "num_domains", "device")}},
            "seg": {"ready": self.seg_model is not None, "error": self.seg_error},
            "sqa": {"ready": self.sqa_bundle is not None, "error": self.sqa_error},
            "engine_mode": "real_pytorch_model" if engine_ok else "dsp_acoustic_fallback",
            "device": str(_DEVICE),
        }


REGISTRY = ModelRegistry()


# ==================================================================== 分类推理
def _center_clip(y: np.ndarray, clip_len: int) -> np.ndarray:
    """与论文 PCGClsDataset._crop_or_pad(is_train=False) 完全一致: 居中裁剪 / 平铺补齐。"""
    if len(y) >= clip_len:
        start = (len(y) - clip_len) // 2
        return y[start:start + clip_len]
    reps = int(np.ceil(clip_len / max(1, len(y))))
    return np.tile(y, reps)[:clip_len]


def extract_dualview(y: np.ndarray, use_cwt: Optional[bool] = None,
                     checkpoint: Optional[str] = None):
    """提取 Log-Mel + CWT 双视图 (纯 DSP, 不依赖模型权重)。

    即使 cls_best.pt 尚未就绪, 前端声像对照模块也需要 mel/cwt, 因此单独暴露。
    去 CWT 的消融模型 (use_cwt=False) 会拿到全零的 cwt, 与训练时口径一致。
    """
    sr = C.AUDIO["target_sr"]
    clip = _center_clip(y, int(sr * C.AUDIO["cls_clip_sec"]))
    mel = extract_log_mel(clip, sr=sr, n_mels=C.AUDIO["n_mels"],
                          n_fft=C.AUDIO["n_fft"], hop_length=C.AUDIO["hop_length"])
    if use_cwt is None:
        use_cwt = bool(REGISTRY.cls_meta(checkpoint).get("use_cwt", True))
    if use_cwt:
        cwt = extract_cwt_scalogram(clip, sr=sr, n_scales=C.AUDIO["cwt_scales"],
                                    target_time_frames=mel.shape[1])
    else:
        cwt = np.zeros_like(mel)
    return clip, mel, cwt


def _enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict_cls(y: np.ndarray, mc_passes: int = C.MC_PASSES,
                checkpoint: Optional[str] = None) -> dict:
    """分类推理 + MC-Dropout 不确定性 (与论文 evaluate_with_uncertainty 同口径)。

    checkpoint 为空时用主模型; 传入对比池里的其它键可对同一片段换模型重算。
    """
    model = REGISTRY.ensure_cls(checkpoint)
    if model is None:
        return {"ready": False, "error": REGISTRY.cls_error(checkpoint),
                "checkpoint": checkpoint or C.DEFAULT_CHECKPOINT}

    clip, mel, cwt = extract_dualview(y, checkpoint=checkpoint)

    xm = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    xc = torch.from_numpy(cwt).unsqueeze(0).unsqueeze(0).to(_DEVICE)

    model.eval()
    out = model(xm, xc, grl_lambda=0.0)
    prob_det = float(torch.softmax(out["logits"], dim=1)[0, 1].item())
    emb = out["embedding"][0].detach().cpu().numpy().astype(float).round(5).tolist()

    probs = [prob_det]
    if mc_passes > 1:
        # MC-Dropout 是随机采样: 不固定种子时同一条录音两次分析可能给出不同的
        # 概率甚至不同的判定 (实测 13918_AV 会在 0.394~0.401 间跳动, 恰好跨过
        # 决策边际门)。临床辅助工具必须可复现, 因此固定种子并在结束后恢复现场。
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(_MC_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_MC_SEED)
            _enable_mc_dropout(model)
            for _ in range(mc_passes):
                o = model(xm, xc, grl_lambda=0.0)
                probs.append(float(torch.softmax(o["logits"], dim=1)[0, 1].item()))
        finally:
            model.eval()
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    arr = np.array(probs, dtype=float)
    p_mean = float(arr.mean())
    p_clip = float(np.clip(p_mean, 1e-6, 1 - 1e-6))
    entropy = -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))
    unc = float(0.5 * entropy + 0.5 * float(arr.std()))

    meta = REGISTRY.cls_meta(checkpoint)
    return {
        "ready": True,
        "checkpoint": checkpoint or C.DEFAULT_CHECKPOINT,
        "prob_abnormal": round(p_mean, 4),
        "prob_normal": round(1 - p_mean, 4),
        "prob_deterministic": round(prob_det, 4),
        "uncertainty": round(unc, 4),
        "mc_passes": len(probs),
        "mc_samples": [round(p, 4) for p in probs],
        "embedding": emb,
        "mel": mel.round(4).tolist(),
        "cwt": cwt.round(4).tolist(),
        "meta": meta,
    }


# ==================================================================== Grad-CAM
def compute_gradcam(y: np.ndarray, checkpoint: Optional[str] = None) -> Optional[dict]:
    """对 enc_mel 的 cbam4 (无 CBAM 时回退 layer4) 做 Grad-CAM，返回热力图矩阵。"""
    model = REGISTRY.ensure_cls(checkpoint)
    if model is None:
        return None
    clip, mel, cwt = extract_dualview(y, checkpoint=checkpoint)

    xm = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    xc = torch.from_numpy(cwt).unsqueeze(0).unsqueeze(0).to(_DEVICE)

    # 定位目标层
    target = None
    if hasattr(model, "enc_mel"):
        m = model.enc_mel.m
        target = m["cbam4"]
        if isinstance(target, torch.nn.Identity):
            target = m["layer4"]
    elif hasattr(model, "features"):
        target = model.features[-1]
    elif hasattr(model, "cnn"):
        target = model.cnn[-2]
    if target is None:
        return None

    store = {}
    h1 = target.register_forward_hook(lambda mod, i, o: store.__setitem__("f", o))
    h2 = target.register_full_backward_hook(lambda mod, gi, go: store.__setitem__("g", go[0]))
    try:
        model.eval()
        # 注意: 必须保留参数 requires_grad=True, 否则 logits 没有 grad_fn, backward 会失败
        for p in model.parameters():
            p.requires_grad_(True)
        out = model(xm, xc, grl_lambda=0.0)
        logits = out["logits"]
        prob = float(torch.softmax(logits, dim=1)[0, 1].item())
        cls = 1 if prob >= 0.5 else 0
        model.zero_grad(set_to_none=True)
        logits[0, cls].backward()

        fmap = store.get("f")
        grad = store.get("g")
        if fmap is None or grad is None:
            return None
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=mel.shape, mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return {"target_class": int(cls), "prob_abnormal": round(prob, 4),
                "checkpoint": checkpoint or C.DEFAULT_CHECKPOINT,
                "cam": cam.round(4).tolist(), "shape": list(cam.shape)}
    except Exception as ex:                                      # noqa: BLE001
        warnings.warn("Grad-CAM failed: %s: %s" % (type(ex).__name__, ex))
        return None
    finally:
        h1.remove(); h2.remove()
        model.eval()


# ==================================================================== 分割推理
@torch.no_grad()
def predict_seg(y: np.ndarray) -> dict:
    """四通道包络 -> 1D U-Net -> 逐帧四状态 -> Viterbi 解码 -> S1/S2 事件时间戳。"""
    model = REGISTRY.ensure_seg()
    if model is None:
        return {"ready": False, "error": REGISTRY.seg_error}

    sr = C.AUDIO["target_sr"]
    env_sr, win = C.AUDIO["env_sr"], C.AUDIO["seg_win_len"]
    envs = compute_4ch_envelopes(y, sr=sr, env_sr=env_sr)      # [4, T]
    T = envs.shape[1]
    if T >= win:
        start = max(0, (T - win) // 2)
        seg = envs[:, start:start + win]
    else:
        seg = np.pad(envs, ((0, 0), (0, win - T)), mode="constant")

    x = torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0).float().to(_DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()       # [4, win]
    seq = viterbi_cardiac_decode(probs)

    s1 = extract_event_centers(seq, 0, env_sr)
    s2 = extract_event_centers(seq, 2, env_sr)
    dur_real = T / float(env_sr)
    return {
        "ready": True,
        "env_sr": env_sr,
        "duration_sec": round(dur_real, 2),
        "frames": int(win),
        "state_names": C.SEG["state_names"],
        "state_seq": seq.astype(int).tolist(),
        "state_probs": probs.argmax(axis=0).astype(int).tolist(),
        "s1_centers_sec": [round(float(t), 3) for t in s1 if t <= dur_real],
        "s2_centers_sec": [round(float(t), 3) for t in s2 if t <= dur_real],
        "envelopes_4ch": envs[:, :min(T, win * 2)].round(4).tolist(),
        "status": "Viterbi 生理周期约束解码完成",
    }


# ==================================================================== SQA 质控
def predict_sqa(feats: dict) -> dict:
    bundle = REGISTRY.ensure_sqa()
    if bundle is None:
        return {"ready": False, "error": REGISTRY.sqa_error}
    # 论文训练时用 DataFrame + 列名拟合 RandomForest, 所以推理也传同名列的 DataFrame,
    # 否则 sklearn 会报 "X does not have valid feature names" 且无法对齐特征顺序。
    import pandas as pd
    vec = pd.DataFrame([[float(feats.get(k, 0.0)) for k in C.SQA_FEATURES]],
                       columns=C.SQA_FEATURES)
    prob = float(bundle["model"].predict_proba(vec)[0, 1])
    imp = {k: round(float(v), 4) for k, v in
           zip(C.SQA_FEATURES, bundle["model"].feature_importances_)}
    return {"ready": True, "prob_quality_pass": round(prob, 4), "feature_importance": imp}


def extract_quality(y_raw: np.ndarray, sr: int) -> dict:
    """六维质控 (直接复用论文 extract_sqa_6d_features, 口径一致) + 可读问题清单。"""
    feats = extract_sqa_6d_features(y_raw, sr=sr)
    issues = []
    if feats["duration_sec"] < 3.0:
        issues.append("录音时长不足 3s，建议延长采集时间")
    if feats["rms_energy"] < 0.008:
        issues.append("信号幅度过低，请检查听诊头与胸壁贴合")
    if feats["silence_ratio"] > 0.35:
        issues.append("静音片段比例过高，请检查拾音连接")
    if feats["clipping_ratio"] > 0.02:
        issues.append("存在峰值削波失真，请调低前端增益")
    if feats["snr_proxy_db"] < 6.0:
        issues.append("背景高频噪声较强，建议换安静环境")
    if feats["stability_cv"] > 1.2:
        issues.append("能量波动过大，请保持受试者静息")

    sqa = predict_sqa(feats)
    if sqa.get("ready"):
        passed = bool(sqa["prob_quality_pass"] >= 0.5 and len(issues) <= 1)
    else:
        passed = len(issues) == 0

    return {
        "features": feats,
        "sqa": sqa,
        "issues": issues,
        "passed": passed,
        "status_text": "采集质量合格" if passed else "质量未达标，建议重采",
    }


def load_audio_from_path(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """返回 (带通滤波并归一化后的 y, 原始 raw_y, sr)"""
    y, raw = load_and_preprocess_audio(path, target_sr=C.AUDIO["target_sr"],
                                       lowcut=C.AUDIO["bandpass_low"],
                                       highcut=C.AUDIO["bandpass_high"])
    return y, raw, C.AUDIO["target_sr"]