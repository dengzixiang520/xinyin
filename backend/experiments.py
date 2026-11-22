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
    fs = raw.get("final_metrics_std", {}) or {}      # 跨折标准差 (5 折实验才有)
    args = raw.get("args", {}) or {}
    hist = raw.get("history", []) or []
    curve = [{"epoch": h.get("epoch"), "macc": h.get("macc_uar"),
              "auroc": h.get("auroc"), "loss": h.get("tr_loss")}
             for h in hist if h.get("epoch") is not None]
    folds = raw.get("folds", []) or []
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
        "std": {k: fs.get(k) for k in
                ("accuracy", "sensitivity", "specificity", "macc_uar",
                 "macro_f1", "auroc", "ece")} if fs else None,
        "n_folds": len(folds) if folds else None,
        "folds": [{"fold": f.get("fold"), "macc_uar": f.get("macc_uar"),
                   "auroc": f.get("auroc"), "sensitivity": f.get("sensitivity"),
                   "specificity": f.get("specificity")} for f in folds] if folds else None,
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
            "aggregated": args.get("aggregated"),
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
                          "available": False, "metrics": {}, "config": {},
                          "curve": [], "std": None, "n_folds": None,
                          "folds": None})

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


# ==================================================================== 消融差值表
# 每个消融项移除的机制 + 中文说明 (与论文 Table 4 一一对应)
ABLATION_MECHANISM = {
    "abl_no_cwt": ("CWT 双视图", "Morlet 连续小波时频视图，退化为仅 Log-Mel 单视图"),
    "abl_no_cbam": ("CBAM 注意力", "通道 + 空间注意力，模型失去频带/时段选择性"),
    "abl_no_mixstyle": ("MixStyle 风格随机化", "特征层风格混合，失去设备响应的风格不变性"),
    "abl_no_grl": ("GRL 域对抗", "梯度反转层压制域可分性，失去显式域对齐"),
    "abl_no_aug": ("肺音/频响增强", "肺音混叠与随机频响扰动，失去采集条件鲁棒性"),
}


def ablation_table() -> dict:
    """Table 4: 各消融项相对完整模型 (cross_proposed_full) 的真实差值。"""
    exps = experiments()
    by = {i["name"]: i for i in exps["items"]}

    full = by.get("cross_proposed_full", {})
    fm = (full.get("metrics") or {}) if full.get("available") else {}
    base_macc = fm.get("macc_uar")
    base_auroc = fm.get("auroc")

    rows = []
    for key, (mech, desc) in ABLATION_MECHANISM.items():
        it = by.get(key, {})
        ok = it.get("available", False)
        m = (it.get("metrics") or {}) if ok else {}
        macc, auroc = m.get("macc_uar"), m.get("auroc")
        rows.append({
            "name": key,
            "label": it.get("label", key),
            "available": ok,
            "mechanism": mech,
            "mechanism_desc": desc,
            "macc_uar": macc,
            "auroc": auroc,
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "delta_macc": (round(base_macc - macc, 4)
                           if base_macc is not None and macc is not None else None),
            "delta_auroc": (round(base_auroc - auroc, 4)
                            if base_auroc is not None and auroc is not None else None),
        })

    vals = [r["delta_macc"] for r in rows if r["delta_macc"] is not None]
    ranked = sorted([r for r in rows if r["delta_macc"] is not None],
                    key=lambda r: -r["delta_macc"]) if vals else []
    return {
        "available": bool(vals),
        "baseline": {"name": "cross_proposed_full",
                     "label": full.get("label", "Proposed Full"),
                     "macc_uar": base_macc, "auroc": base_auroc,
                     "sensitivity": fm.get("sensitivity"),
                     "specificity": fm.get("specificity")},
        "rows": rows,
        "n_done": len(vals),
        "n_total": len(rows),
        "most_important": ranked[0]["mechanism"] if ranked else None,
        "note": ("delta = 完整模型 − 消融项, 正值表示该机制被移除后性能下降, "
                 "即该机制确有贡献; 接近 0 表示在本次数据规模下贡献不显著, "
                 "负值需如实报告。"),
    }


# ==================================================================== 数据审计 / 血缘
LICENSE_NOTE = {
    "PhysioNet_2016": "PhysioNet/CinC Challenge 2016，公开数据集",
    "CirCor_2022": "CirCor DigiScope，受控访问 (credentialed)，需 PhysioNet 认证",
    "ZCHSound_2024": "医院合作数据，仅限研究用途，不可再分发",
    "BMD_HS_2024": "公开心音数据库",
    "Fraiwan_2025": "公开心音数据库",
    "HFrEF_2025": "Zenodo 公开数据集",
    "PASCAL_A": "PASCAL Challenge 公开数据集",
    "HLS_CMDS_2025": "公开肺音数据库 (仅作干扰噪声池)",
}

ROLE_NOTE = {
    "PhysioNet_2016": ["跨库训练", "LODO 留一子库测试"],
    "CirCor_2022": ["跨库训练", "分割模型训练"],
    "ZCHSound_2024": ["零样本外测（完全不参与训练）"],
    "BMD_HS_2024": ["跨库训练"],
    "Fraiwan_2025": ["跨库训练"],
    "HFrEF_2025": ["单库 5 折内部验证"],
    "PASCAL_A": ["纳入主清单，未进入主实验"],
    "HLS_CMDS_2025": ["肺音干扰噪声池（增强用）"],
}

TRAIN_DATASETS = ["PhysioNet_2016", "CirCor_2022", "BMD_HS_2024", "Fraiwan_2025"]
TEST_DATASETS = ["ZCHSound_2024"]


def _manifest_checks() -> tuple[list, dict]:
    """在本地清单上做**可复现**的一致性检查 (不依赖服务器文件系统)。

    只声明清单本身能证明的事: 受试者级隔离、标签完整性、时长有效性、
    字段填充率。文件字节级完整性以 data_audit_report 的 corrupt_wavs 为准。
    """
    p = os.path.join(C.RESULTS_DIR, "master_pcg_manifest.csv")
    if not os.path.exists(p):
        return [], {}
    try:
        import pandas as pd
        df = pd.read_csv(p)
    except Exception:                                            # noqa: BLE001
        return [], {}

    n = len(df)
    checks = []
    detail = {}

    # 1) 外测集受试者是否泄漏进训练集
    train_subj = set(df[df["dataset"].isin(TRAIN_DATASETS)]["subject_id"].astype(str))
    test_subj = set(df[df["dataset"].isin(TEST_DATASETS)]["subject_id"].astype(str))
    leak = train_subj & test_subj
    checks.append({"name": "外测集受试者隔离",
                   "detail": f"ZCHSound {len(test_subj)} 名受试者与训练集交集 {len(leak)} 人",
                   "passed": len(leak) == 0})
    detail["test_subjects"] = len(test_subj)
    detail["train_subjects"] = len(train_subj)
    detail["leaked_subjects"] = len(leak)

    # 2) 同一受试者是否跨多个内部折 (防泄漏切分的核心约束)
    #    fold_internal = -1 是「未分配折」的哨兵值 (不参与交叉验证), 必须先剔除,
    #    否则会把 excluded 记录误判成跨折泄漏。
    if "fold_internal" in df.columns:
        assigned = df[df["fold_internal"].notna() & (df["fold_internal"] >= 0)]
        unassigned = int((df["fold_internal"] == -1).sum())
        g = assigned.groupby(["dataset", "subject_id"])["fold_internal"].nunique()
        multi = int((g > 1).sum())
        checks.append({"name": "受试者级折内唯一",
                       "detail": (f"{len(g)} 名受试者跨折 {multi} 人"
                                  f"（另有 {unassigned} 条未分配折，不参与交叉验证）"),
                       "passed": multi == 0})
        detail["subjects_multi_fold"] = multi
        detail["records_unassigned_fold"] = unassigned

    # 3) 标签完整性
    if "label_binary_acoustic" in df.columns:
        miss = int(df["label_binary_acoustic"].isna().sum())
        checks.append({"name": "声学标签完整性",
                       "detail": f"缺失 {miss} / {n} 条",
                       "passed": miss == 0})
        detail["missing_labels"] = miss

    # 4) 时长有效性
    if "duration_sec" in df.columns:
        bad = int((df["duration_sec"].fillna(0) <= 0).sum())
        checks.append({"name": "时长字段有效性",
                       "detail": f"非正时长 {bad} / {n} 条",
                       "passed": bad == 0})

    # 5) 路径字段填充
    if "rel_path" in df.columns:
        miss = int(df["rel_path"].isna().sum())
        checks.append({"name": "音频路径完整性",
                       "detail": f"缺失 {miss} / {n} 条",
                       "passed": miss == 0})

    return checks, detail


def audit() -> dict:
    """数据可追溯性面板: 每个库的来源、许可、实测条数、参与角色 + 可复现检查。"""
    d = _read_json(os.path.join(C.RESULTS_DIR, "data_audit_report.json")) or {}
    man = _manifest_totals()
    raw = d.get("datasets", {}) or {}
    per = {r["dataset"]: r for r in (man.get("per_dataset") or [])}

    rows = []
    for name in sorted(set(list(raw.keys()) + list(per.keys()))):
        v = raw.get(name, {}) or {}
        p = per.get(name, {}) or {}
        rows.append({
            "dataset": name,
            "recordings": v.get("recordings") or p.get("records"),
            "subjects": v.get("unique_subjects"),
            "hours": p.get("hours"),
            "mean_sec": p.get("mean_sec"),
            "n_domains": len(v.get("domains", {}) or {}),
            "domains": v.get("domains", {}) or {},
            "sample_rates": v.get("sample_rates", {}) or {},
            "duration_min_med_max": v.get("duration_sec_min_med_max"),
            "license": LICENSE_NOTE.get(name, "未登记"),
            "roles": ROLE_NOTE.get(name, []),
            "used_in_training": name in TRAIN_DATASETS,
            "is_holdout": name in TEST_DATASETS,
            "label_acoustic": v.get("label_binary_acoustic", {}) or {},
        })
    rows.sort(key=lambda r: -(r["recordings"] or 0))

    corrupt = d.get("corrupt_wavs", {}) or {}
    total_corrupt = int(sum(int(x) for x in corrupt.values())) if corrupt else 0

    checks, detail = _manifest_checks()
    # 字节级完整性以服务器端全量扫描结果为准
    checks.insert(0, {"name": "音频文件可解码性",
                      "detail": f"损坏文件 {total_corrupt} 个（服务器端全量扫描）",
                      "passed": total_corrupt == 0})

    return {
        "available": bool(rows),
        "datasets": rows,
        "n_datasets": len(rows),
        "total_records": man.get("total_records"),
        "total_hours": man.get("total_hours"),
        "total_subjects": man.get("total_subjects"),
        "n_domains": man.get("n_domains"),
        "corrupt_wavs": corrupt,
        "corrupt_total": total_corrupt,
        "lung_noise_pool_count": d.get("lung_noise_pool_count"),
        "checks": checks,
        "checks_detail": detail,
        "n_checks_passed": sum(1 for c in checks if c["passed"]),
        "n_checks": len(checks),
    }