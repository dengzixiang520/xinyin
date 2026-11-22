# -*- coding: utf-8 -*-
"""
声见其迹 — 论文实验结果读取层

比赛端展示的所有数字 **一律来自论文真实训练产物**, 不做任何硬编码或伪造。
数据源 (由 sync_results.py 从 3090 服务器同步):

  results/sqa_results.json          Stage1 六维 SQA 质控标定 (5折 + 跨库零样本)
  results/seg_history_fold0.json    Stage2 1D U-Net 分割训练曲线 + 最佳事件F1
  results/data_audit_report.json    Table 1 数据集统计
  results/experiments/<name>.json   Stage3/4 每个分类实验的 metrics.json

未跑完的实验在接口里以 available=false 出现, 前端据此显示 "训练中"。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

from . import config as C

EXPERIMENTS_DIR = os.path.join(C.RESULTS_DIR, "experiments")

# 论文 Table 2/3/4 的实验顺序与中文标签 (仅用于展示排序, 不参与计算)
EXP_ORDER = [
    ("lodo_baseline_cnn",     "Table2", "LODO · Baseline CNN (无域泛化)"),
    ("lodo_proposed_dg",      "Table2", "LODO · Proposed DG-ResNet-CBAM"),
    ("cross_single_pn16_to_zch", "Table3", "跨库 · 仅 PhysioNet→ZCHSound"),
    ("cross_naive_pool_cnn",  "Table3", "跨库 · 多源普通混训 CNN"),
    ("cross_crnn_cbam",       "Table3", "跨库 · CRNN-CBAM"),
    ("cross_proposed_full",   "Table3", "跨库 · Proposed Full (本系统)"),
    ("abl_no_cwt",            "Table4", "消融 · 去 CWT 双视图"),
    ("abl_no_cbam",           "Table4", "消融 · 去 CBAM 注意力"),
    ("abl_no_mixstyle",       "Table4", "消融 · 去 MixStyle 风格随机化"),
    ("abl_no_grl",            "Table4", "消融 · 去 GRL 域对抗"),
    ("abl_no_aug",            "Table4", "消融 · 去 肺音/频响增强"),
    ("internal_hfref_5fold",  "Table5", "单库 5 折 · HFrEF 2025"),
]


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                            # noqa: BLE001
        return None


def sqa_summary() -> dict:
    d = _read_json(os.path.join(C.RESULTS_DIR, "sqa_results.json"))
    if not d:
        return {"available": False}
    return {"available": True, **d,
            "feature_labels": C.SQA_FEATURE_LABELS}


def seg_summary() -> dict:
    d = _read_json(os.path.join(C.RESULTS_DIR, "seg_history_fold0.json"))
    if not d:
        return {"available": False}
    hist = d.get("history", [])
    curve = [{"epoch": h["epoch"], "frame_acc": h.get("frame_acc"),
              "event_f1": h.get("event_f1"), "loss": h.get("tr_loss")} for h in hist]
    best = d.get("best_event_f1")
    best_ep = None
    if curve and best is not None:
        cand = [c for c in curve if c.get("event_f1") is not None]
        if cand:
            best_ep = max(cand, key=lambda x: x["event_f1"])["epoch"]
    return {"available": True, "best_event_f1": best, "best_epoch": best_ep,
            "epochs": len(curve), "curve": curve,
            "final_frame_acc": curve[-1].get("frame_acc") if curve else None}


def _manifest_totals() -> dict:
    """从 master_pcg_manifest.csv 计算精确总时长 (data_audit_report 只给了 min/med/max)。"""
    p = os.path.join(C.RESULTS_DIR, "master_pcg_manifest.csv")
    if not os.path.exists(p):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(p)
        total_sec = float(df["duration_sec"].sum())
        by = (df.groupby("dataset")["duration_sec"]
                .agg(["count", "sum", "mean"]).reset_index())
        return {
            "total_records": int(len(df)),
            "total_hours": round(total_sec / 3600.0, 2),
            "total_subjects": int(df["subject_id"].nunique()),
            "n_domains": int(df["domain_id"].nunique()),
            "per_dataset": [
                {"dataset": r["dataset"], "records": int(r["count"]),
                 "hours": round(float(r["sum"]) / 3600.0, 2),
                 "mean_sec": round(float(r["mean"]), 1)}
                for _, r in by.sort_values("sum", ascending=False).iterrows()
            ],
        }
    except Exception:                                            # noqa: BLE001
        return {}


def dataset_table() -> dict:
    d = _read_json(os.path.join(C.RESULTS_DIR, "data_audit_report.json"))
    man = _manifest_totals()
    if not d:
        return {"available": bool(man), **man}

    raw = d.get("datasets", {}) or {}
    rows = []
    for name, v in raw.items():
        rows.append({
            "dataset": name,
            "recordings": v.get("recordings"),
            "subjects": v.get("unique_subjects"),
            "domains": v.get("domains", {}),
            "sample_rates": v.get("sample_rates", {}),
            "duration_min_med_max": v.get("duration_sec_min_med_max"),
            "label_acoustic": v.get("label_binary_acoustic", {}),
            "label_clinical": v.get("label_binary_clinical", {}),
            "sqi": v.get("sqi_label_good1_poor0", {}),
            "folds": v.get("fold_distribution", {}),
        })
    rows.sort(key=lambda r: -(r["recordings"] or 0))

    corrupt = d.get("corrupt_wavs", {}) or {}
    return {
        "available": True,
        "datasets": rows,
        "n_datasets": len(rows),
        "lung_noise_pool_count": d.get("lung_noise_pool_count"),
        "corrupt_wavs": corrupt,
        "corrupt_total": int(sum(int(x) for x in corrupt.values())) if corrupt else 0,
        **man,
    }


def _extract_metrics(raw: dict) -> dict:
    """从论文 metrics.json 结构里抽取前端要用的扁平字段。"""
    fm = raw.get("final_metrics", {}) or {}
    args = raw.get("args", {}) or {}
    hist = raw.get("history", []) or []
    curve = [{"epoch": h.get("epoch"), "macc": h.get("macc_uar"),
              "auroc": h.get("auroc"), "loss": h.get("tr_loss")}
             for h in hist if h.get("epoch") is not None]
    return {
        "metrics": {
            "accuracy": fm.get("accuracy"),
            "sensitivity": fm.get("sensitivity"),
            "specificity": fm.get("specificity"),
            "macc_uar": fm.get("macc_uar"),
            "macro_f1": fm.get("macro_f1"),
            "auroc": fm.get("auroc"),
            "ece": fm.get("ece"),
            "params_M": fm.get("params_M"),
            "latency_ms_5s_clip": fm.get("latency_ms_5s_clip"),
            "selective_abstention": fm.get("selective_abstention"),
            "confusion_matrix": fm.get("confusion_matrix"),
        },
        "config": {
            "protocol": args.get("protocol"),
            "model": args.get("model"),
            "test_domain": args.get("test_domain"),
            "test_dataset": args.get("test_dataset"),
            "train_datasets": args.get("train_datasets"),
            "no_cwt": args.get("no_cwt"),
            "no_cbam": args.get("no_cbam"),
            "no_mixstyle": args.get("no_mixstyle"),
            "no_grl": args.get("no_grl"),
            "no_aug": args.get("no_aug"),
            "epochs": args.get("epochs"),
        },
        "curve": curve,
    }


def experiments() -> dict:
    """返回全部论文实验 (含未完成项) 的真实指标。"""
    found = {}
    for name in glob.glob(os.path.join(EXPERIMENTS_DIR, "*.json")):
        base = os.path.splitext(os.path.basename(name))[0]
        raw = _read_json(name)
        if raw:
            found[base] = _extract_metrics(raw)

    items = []
    for key, table, label in EXP_ORDER:
        if key in found:
            items.append({"name": key, "table": table, "label": label,
                          "available": True, **found[key]})
        else:
            items.append({"name": key, "table": table, "label": label,
                          "available": False, "metrics": {}, "config": {}, "curve": []})

    # 期刊/会议表里需要的对比摘要
    def m(name, field):
        v = found.get(name, {}).get("metrics", {}).get(field)
        return v

    headline = {
        "proposed_macc": m("cross_proposed_full", "macc_uar"),
        "proposed_auroc": m("cross_proposed_full", "auroc"),
        "baseline_macc": m("cross_naive_pool_cnn", "macc_uar"),
        "single_macc": m("cross_single_pn16_to_zch", "macc_uar"),
        "lodo_base_macc": m("lodo_baseline_cnn", "macc_uar"),
        "lodo_prop_macc": m("lodo_proposed_dg", "macc_uar"),
    }
    if headline["proposed_macc"] and headline["baseline_macc"]:
        headline["gain_over_naive"] = round(
            headline["proposed_macc"] - headline["baseline_macc"], 4)
    else:
        headline["gain_over_naive"] = None
    if headline["lodo_prop_macc"] and headline["lodo_base_macc"]:
        headline["gain_lodo"] = round(
            headline["lodo_prop_macc"] - headline["lodo_base_macc"], 4)
    else:
        headline["gain_lodo"] = None

    n_done = sum(1 for i in items if i["available"])
    return {"available": n_done > 0, "total": len(items), "completed": n_done,
            "items": items, "headline": headline}


def overview() -> dict:
    """仪表盘首页需要的聚合数字。"""
    sqa = sqa_summary()
    seg = seg_summary()
    exps = experiments()
    audit = dataset_table()

    total_records = audit.get("total_records")
    total_hours = audit.get("total_hours")
    total_subjects = audit.get("total_subjects")

    return {
        "sqa": {"available": sqa.get("available"),
                "cv_auc": sqa.get("5fold_mean_auc"),
                "cv_auc_std": sqa.get("5fold_std_auc"),
                "cv_f1": sqa.get("5fold_mean_macro_f1"),
                "zero_shot_auc": sqa.get("zero_shot_zchsound_auc")},
        "seg": {"available": seg.get("available"),
                "best_event_f1": seg.get("best_event_f1"),
                "final_frame_acc": seg.get("final_frame_acc")},
        "experiments": {"total": exps["total"], "completed": exps["completed"],
                        "headline": exps["headline"]},
        "dataset": {"available": audit.get("available"),
                    "total_records": total_records, "total_hours": total_hours,
                    "total_subjects": total_subjects,
                    "n_datasets": audit.get("n_datasets")},
    }