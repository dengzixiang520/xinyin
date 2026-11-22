# -*- coding: utf-8 -*-
"""
声见其迹 — 风险-覆盖率 / 选择性预测分析层

比赛端最有说服力的一块: 不只报「准确率多少」, 而是回答
「模型在什么情况下不该给结论」。全部曲线由论文真实测试集预测文件
`results/predictions/<experiment>.csv` 逐条重算, 不画示意图。

预测文件字段 (由论文 train_cls.py 导出):
    record_uid, label, domain_idx, prob_abnormal, uncertainty

核对记录: 在 cross_proposed_full 上重算得到的
    coverage 90% -> MAcc 0.6775
    coverage 80% -> MAcc 0.7050
与论文 metrics.json 的 selective_abstention(reject_10pct_macc=0.6775 /
reject_20pct_macc=0.7050) 逐位一致, 说明本模块的拒识口径与论文实现相同。

两道互补的门控:
    1) 认知不确定度门  uncertainty >= thr      —— MC-Dropout 反复采样仍不一致
    2) 决策边际门      |prob - 0.5| < margin   —— 概率贴着决策边界, 等价于掷硬币
在真实测试集上两道门控都单调改善 MAcc, 见 curves()。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import config as C

PRED_DIR = os.path.join(C.RESULTS_DIR, "predictions")
DOMAIN_MAP_PATH = os.path.join(C.RESULTS_DIR, "domain_map.json")

# 三个可选拒识档位。阈值不在代码里写死, 而是按目标拒识率在真实预测分布上反解,
# 因此每个档位显示的覆盖率/准确率都是这台模型在 ZCHSound 外测集上测出来的。
MODE_SPEC = {
    "loose": {"label": "宽松", "target_reject": 0.10, "margin": 0.05,
              "note": "高覆盖率优先, 仅在明显不确定时拒识"},
    "standard": {"label": "标准", "target_reject": 0.20, "margin": 0.10,
                 "note": "论文推荐操作点, 对应 Selective 20% 拒识"},
    "strict": {"label": "严格", "target_reject": 0.30, "margin": 0.15,
               "note": "高风险场景, 覆盖率最低但准确率最高"},
}
DEFAULT_MODE = "standard"

DOMAIN_LABELS = {
    "bmd_littmann3200": "BMD · Littmann 3200",
    "circor_digiscope": "CirCor · DigiScope",
    "fraiwan_littmann3200": "Fraiwan · Littmann 3200",
    "hfref_liyuan_ext": "HFrEF · 丽 Yuan 外院",
    "hfref_tongji": "HFrEF · 同济",
    "pascal_iphone": "PASCAL · iPhone",
    "zch_clean": "ZCHSound · 清洁采集",
    "zch_noisy": "ZCHSound · 噪声采集",
}

_CACHE: dict[str, Any] = {}


# ==================================================================== 基础读取
def _domain_map() -> dict:
    try:
        with open(DOMAIN_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                            # noqa: BLE001
        return {"idx_to_domain": {}, "domain_to_idx": {}}


def available() -> list:
    """列出所有存在真实预测文件的实验名。"""
    if not os.path.isdir(PRED_DIR):
        return []
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(PRED_DIR, "*.csv")))


def load_predictions(name: str) -> Optional[pd.DataFrame]:
    """读取并缓存某实验的逐条预测 (含 label / prob / uncertainty / domain_idx)。"""
    if not name or os.path.basename(name) != name:
        return None                                              # 拒绝路径穿越
    if name in _CACHE:
        return _CACHE[name]
    path = os.path.join(PRED_DIR, name + ".csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if not {"label", "prob_abnormal", "uncertainty"}.issubset(df.columns):
            return None
        df["label"] = df["label"].astype(int)
        df["prob_abnormal"] = df["prob_abnormal"].astype(float)
        df["uncertainty"] = df["uncertainty"].astype(float)
        if "domain_idx" in df.columns:
            df["domain_idx"] = df["domain_idx"].astype(int)
        _CACHE[name] = df
        return df
    except Exception:                                            # noqa: BLE001
        return None


# ==================================================================== 指标计算
def _r(x, nd: int = 4) -> Optional[float]:
    try:
        return round(float(x), nd)
    except Exception:                                            # noqa: BLE001
        return None


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """从混淆矩阵算论文口径的指标 (MAcc = (Se+Sp)/2 即 UAR)。"""
    n = int(len(y))
    if n == 0:
        return {"n": 0, "macc": None, "sensitivity": None, "specificity": None,
                "accuracy": None, "missed": None, "positives": None,
                "negatives": None, "confusion_matrix": None}
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": n,
        "macc": _r((se + sp) / 2.0),
        "sensitivity": _r(se),
        "specificity": _r(sp),
        "accuracy": _r((tp + tn) / n),
        "missed": fn,
        "positives": tp + fn,
        "negatives": tn + fp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _gate(df: pd.DataFrame, thr: Optional[float],
          margin: Optional[float]) -> pd.Series:
    """返回「保留 (给结论)」掩码: 不确定度低于阈值 且 决策边际够大。"""
    keep = pd.Series(True, index=df.index)
    if thr is not None:
        keep &= df["uncertainty"] < float(thr)
    if margin is not None and margin > 0:
        keep &= (df["prob_abnormal"] - 0.5).abs() >= float(margin)
    return keep


def threshold_for_reject_rate(df: pd.DataFrame, rate: float) -> Optional[float]:
    """反解不确定度阈值, 使「按不确定度拒识」的比例恰好等于 rate。

    拒识的是不确定度最高的那一批, 所以阈值取升序序列的第 (n - k) 个,
    而不是第 k 个 (取错方向会把覆盖率压到 rate 本身)。
    """
    if rate <= 0 or rate >= 1:
        return None
    u = np.sort(df["uncertainty"].values)
    n = len(u)
    keep_n = int(round((1.0 - rate) * n))
    keep_n = max(1, min(keep_n, n - 1))
    return _r(u[keep_n])


# ==================================================================== 档位 (给前端)
def modes(name: str) -> list:
    """三个拒识档位, 每个都带在真实测试集上测得的覆盖率与 MAcc。"""
    df = load_predictions(name)
    if df is None:
        return []
    y = df["label"].values
    pred = (df["prob_abnormal"].values >= 0.5).astype(int)
    full = _metrics(y, pred)

    out = []
    for key, spec in MODE_SPEC.items():
        thr = threshold_for_reject_rate(df, spec["target_reject"])
        margin = spec["margin"]
        keep = _gate(df, thr, margin)
        m = _metrics(y[keep.values], pred[keep.values])
        out.append({
            "mode": key,
            "label": spec["label"],
            "note": spec["note"],
            "uncertainty_threshold": thr,
            "margin_threshold": margin,
            "coverage": _r(keep.mean()),
            "abstain_rate": _r(1.0 - keep.mean()),
            "metrics": m,
            "calibrated_on": f"{name} 零样本外测集 (n={len(df)})",
            "gain_macc": (_r(m["macc"] - full["macc"])
                          if m["macc"] is not None and full["macc"] is not None else None),
        })
    return out


def default_gate(name: str = "cross_proposed_full") -> dict:
    """运行期默认门控参数。取自 standard 档位; 无预测文件时退回保守缺省。"""
    spec = MODE_SPEC[DEFAULT_MODE]
    df = load_predictions(name)
    if df is None:
        # 无标定数据时用论文里 uncertainty 分布的中位数附近作保守缺省
        return {"mode": DEFAULT_MODE, "uncertainty_threshold": 0.35,
                "margin_threshold": spec["margin"], "calibrated": False,
                "calibrated_on": None}
    return {"mode": DEFAULT_MODE,
            "uncertainty_threshold": threshold_for_reject_rate(df, spec["target_reject"]),
            "margin_threshold": spec["margin"], "calibrated": True,
            "calibrated_on": f"{name} 零样本外测集 (n={len(df)})"}


def gate_for_mode(name: str, mode: str) -> dict:
    """按指定档位取门控参数 (供前端下拉切换)。"""
    if mode not in MODE_SPEC:
        mode = DEFAULT_MODE
    spec = MODE_SPEC[mode]
    df = load_predictions(name)
    thr = threshold_for_reject_rate(df, spec["target_reject"]) if df is not None else 0.35
    return {"mode": mode, "label": spec["label"],
            "uncertainty_threshold": thr, "margin_threshold": spec["margin"],
            "calibrated": df is not None,
            # 三个档位都来自同一份标定数据, 都要能说清阈值出处
            "calibrated_on": (f"{name} 零样本外测集 (n={len(df)})" if df is not None
                              else None)}


# ==================================================================== 曲线
def curves(name: str, step: float = 0.05) -> dict:
    """三条曲线, 全部由真实预测逐点重算。

    1) uncertainty: 只按不确定度拒识 (论文 selective_abstention 口径)
    2) margin:      只按决策边际拒识
    3) dual:        两者同时启用 (standard 档位)
    """
    df = load_predictions(name)
    if df is None:
        return {"available": False, "experiment": name}

    y = df["label"].values
    p = df["prob_abnormal"].values
    u = df["uncertainty"].values
    pred = (p >= 0.5).astype(int)
    n = len(y)

    # ---- 不确定度曲线: 按不确定性从高到低逐个拒识
    order_u = np.argsort(-u)
    # ---- 边际曲线: 按 |p-0.5| 从小到大逐个拒识
    order_m = np.argsort(np.abs(p - 0.5))

    def _sweep(order: np.ndarray) -> list:
        pts = []
        k = 0
        while k < n:
            keep = np.ones(n, dtype=bool)
            keep[order[:k]] = False
            m = _metrics(y[keep], pred[keep])
            cov = float(keep.sum()) / n
            pts.append({"coverage": _r(cov), "abstain_rate": _r(1 - cov),
                        "macc": m["macc"], "sensitivity": m["sensitivity"],
                        "specificity": m["specificity"], "kept": m["n"]})
            k = max(k + 1, int(round((k / n + step) * n)))
        pts.append({"coverage": _r(1.0 / n), "abstain_rate": _r(1 - 1.0 / n),
                    "macc": None, "sensitivity": None, "specificity": None, "kept": 1})
        return pts

    full = _metrics(y, pred)
    mode_std = next((m for m in modes(name) if m["mode"] == DEFAULT_MODE), None)
    thr = mode_std["uncertainty_threshold"] if mode_std else None
    mg = mode_std["margin_threshold"] if mode_std else None
    keep_dual = _gate(df, thr, mg)
    dual = _metrics(y[keep_dual.values], pred[keep_dual.values])

    return {
        "available": True,
        "experiment": name,
        "n_test": n,
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "uncertainty_range": {"min": _r(u.min()), "median": _r(np.median(u)),
                              "p90": _r(np.percentile(u, 90)),
                              "p95": _r(np.percentile(u, 95)), "max": _r(u.max())},
        "no_gate": full,
        "operating_point": {"mode": DEFAULT_MODE,
                            "uncertainty_threshold": thr, "margin_threshold": mg,
                            "coverage": _r(keep_dual.mean()), **dual},
        "curve_uncertainty": _sweep(order_u),
        "curve_margin": _sweep(order_m),
        "modes": modes(name),
    }


def calibration(name: str, bins: int = 10) -> dict:
    """可靠度图 (reliability diagram) + 期望校准误差 ECE。"""
    df = load_predictions(name)
    if df is None:
        return {"available": False}
    y = df["label"].values.astype(int)
    p = df["prob_abnormal"].values
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)          # 预测类别的置信度

    edges = np.linspace(0.5, 1.0, bins + 1)
    rows = []
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        acc = float((y[sel] == pred[sel]).mean())
        mean_conf = float(conf[sel].mean())
        ece += (cnt / n) * abs(acc - mean_conf)
        rows.append({"bin": [float(lo), float(hi)], "count": cnt,
                     "share": _r(cnt / n), "mean_confidence": _r(mean_conf),
                     "empirical_accuracy": _r(acc), "gap": _r(acc - mean_conf)})
    # 分「被判为某类」的样本看校准方向: 判为异常的那批里真正异常占多少
    sep = {}
    s_pos = pred == 1
    if s_pos.sum():
        sep["abnormal"] = {"n": int(s_pos.sum()),
                           "mean_prob": _r(p[s_pos].mean()),
                           "empirical_rate": _r((y[s_pos] == 1).mean())}
    s_neg = pred == 0
    if s_neg.sum():
        sep["normal"] = {"n": int(s_neg.sum()),
                         "mean_prob": _r((1.0 - p[s_neg]).mean()),
                         "empirical_rate": _r((y[s_neg] == 0).mean())}
    return {"available": True, "experiment": name, "n": n, "bins": bins,
            "ece": _r(ece), "reliability": rows, "per_class": sep}


def domain_breakdown(name: str, with_gate: bool = True) -> dict:
    """按设备/站点域分解性能 —— 域泛化能力最直接的证据。"""
    df = load_predictions(name)
    if df is None or "domain_idx" not in df.columns:
        return {"available": False, "experiment": name}
    idx2dom = _domain_map().get("idx_to_domain", {})
    y = df["label"].values
    p = df["prob_abnormal"].values
    pred = (p >= 0.5).astype(int)
    gate = default_gate(name)
    keep = _gate(df, gate["uncertainty_threshold"], gate["margin_threshold"]).values

    rows = []
    for di in sorted(df["domain_idx"].unique().tolist()):
        sel = (df["domain_idx"] == di).values
        dom = idx2dom.get(str(di), f"domain_{di}")
        base = _metrics(y[sel], pred[sel])
        keep_sel = sel & keep
        gated = _metrics(y[keep_sel], pred[keep_sel])
        # 该域内开拒识后的实际覆盖率 (前端直接用, 不用自己拿 n 相除)
        gated["coverage"] = _r(keep_sel.sum() / sel.sum()) if sel.sum() else None
        gated["abstained"] = int(sel.sum() - keep_sel.sum())
        rows.append({
            "domain_idx": int(di),
            "domain": dom,
            "label": DOMAIN_LABELS.get(dom, dom),
            "share": _r(sel.sum() / len(df)),
            "no_gate": base,
            "with_gate": gated if with_gate else None,
        })
    rows.sort(key=lambda r: -(r["no_gate"]["n"] or 0))
    all_m = _metrics(y, pred)
    if len(rows) >= 2:
        maccs = [r["no_gate"]["macc"] for r in rows if r["no_gate"]["macc"] is not None]
        spread = _r(max(maccs) - min(maccs)) if maccs else None
    else:
        spread = None
    return {"available": True, "experiment": name, "overall": all_m,
            "domains": rows, "worst_minus_best": spread,
            "gate_used": gate}


def summary(name: str = "cross_proposed_full") -> dict:
    """前端一次拿全: 门控档位 + 曲线 + 校准 + 域分解。"""
    df = load_predictions(name)
    if df is None:
        return {"available": False, "experiment": name,
                "available_experiments": available()}
    return {"available": True, "experiment": name,
            "available_experiments": available(),
            "gate": default_gate(name),
            "curves": curves(name),
            "calibration": calibration(name),
            "domain": domain_breakdown(name)}