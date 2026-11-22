#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声见其迹 SONOTRACE — 竞赛信息表导出工具

把后端从论文真实训练产物里读出来的表格，导出成 CSV 放到 `docs/tables/`，
方便评委/队友直接拿去做 Excel 或贴进材料，**不用手抄数字**。

所有数值的唯一来源仍是 `results/` 下的产物，本脚本只做"搬运 + 排版"，
不做任何计算补全：接口里是 None 的字段导出为空单元格，不填 0、不估算。

用法:
    cd sonotrace-web
    python tools/export_tables.py
    python tools/export_tables.py --out docs/tables
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend import experiments as EXP      # noqa: E402
from backend import selective as SEL        # noqa: E402

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(LOCAL_ROOT, "docs", "tables")


def _cell(v):
    """None -> 空字符串; 浮点保留 4 位; 其余原样。"""
    if v is None:
        return ""
    if isinstance(v, float):
        return "%.4f" % v
    return str(v)


def _write(path: str, header: list, rows: list) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_cell(x) for x in r])
    return len(rows)


# ==================================================================== 各张表
def table1_datasets() -> tuple:
    a = EXP.audit()
    header = ["数据集", "记录数", "受试者", "时长(h)", "平均时长(s)",
              "设备域数", "采样率(Hz)", "许可证", "角色", "参与训练", "零样本外测"]
    rows = []
    for r in a["datasets"]:
        rows.append([r["dataset"], r["recordings"], r["subjects"], r["hours"],
                     r["mean_sec"], r["n_domains"],
                     "/".join((r["sample_rates"] or {}).keys()),
                     r["license"], "；".join(r["roles"]),
                     "是" if r["used_in_training"] else "否",
                     "是" if r["is_holdout"] else "否"])
    return header, rows, "数据集统计"


def table23_models() -> tuple:
    d = EXP.experiments()
    by = {i["name"]: i for i in d["items"]}
    order = ["lodo_baseline_cnn", "lodo_proposed_dg",
             "cross_single_pn16_to_zch", "cross_naive_pool_cnn",
             "cross_crnn_cbam", "cross_proposed_full", "internal_hfref_5fold"]
    header = ["分组", "实验", "MAcc/UAR", "AUROC", "敏感度", "特异度",
              "Macro-F1", "参数量(M)", "时延(ms/5s)", "MAcc标准差", "AUROC标准差", "5折数"]
    rows = []
    for n in order:
        it = by.get(n)
        if not it:
            continue
        m = it.get("metrics") or {}
        s = it.get("std") or {}
        rows.append([it.get("table"), it.get("label"), m.get("macc_uar"),
                     m.get("auroc"), m.get("sensitivity"), m.get("specificity"),
                     m.get("macro_f1"), m.get("params_M"),
                     m.get("latency_ms_5s_clip"),
                     s.get("macc_uar"), s.get("auroc"), it.get("n_folds")])
    return header, rows, "Table2-3-5_模型对比"


def table4_ablation() -> tuple:
    t = EXP.ablation_table()
    header = ["移除的机制", "说明", "MAcc", "ΔMAcc", "ΔAUROC", "敏感度", "特异度", "完成"]
    rows = []
    for r in t["rows"]:
        rows.append([r["mechanism"], r["mechanism_desc"], r["macc_uar"],
                     r["delta_macc"], r["delta_auroc"], r["sensitivity"],
                     r["specificity"], "是" if r["available"] else "否"])
    return header, rows, "消融实验"


def table5_folds() -> tuple:
    d = EXP.experiments()
    by = {i["name"]: i for i in d["items"]}
    it = by.get("internal_hfref_5fold") or {}
    header = ["折", "MAcc/UAR", "AUROC", "敏感度", "特异度"]
    rows = []
    for f in (it.get("folds") or []):
        rows.append([f.get("fold"), f.get("macc_uar"), f.get("auroc"),
                     f.get("sensitivity"), f.get("specificity")])
    return header, rows, "内部5折_逐折"


def table6_rejection() -> tuple:
    exp = "cross_proposed_full"
    modes = SEL.modes(exp)
    header = ["档位", "不确定度阈值", "边际阈值", "覆盖率", "拒识率",
              "MAcc", "相对不拒识增益", "敏感度", "特异度", "标定来源"]
    rows = []
    for m in modes:
        mm = m["metrics"]
        rows.append([m["label"], m["uncertainty_threshold"], m["margin_threshold"],
                     m["coverage"], m["abstain_rate"], mm.get("macc"),
                     m["gain_macc"], mm.get("sensitivity"), mm.get("specificity"),
                     m.get("calibrated_on")])
    return header, rows, "风险覆盖率_三档"


def table7_domain() -> tuple:
    dd = SEL.domain_breakdown("cross_proposed_full")
    header = ["设备域", "占比", "不拒识MAcc", "不拒识敏感度", "不拒识特异度",
              "拒识后MAcc", "拒识后覆盖率", "样本数"]
    rows = []
    for r in dd["domains"]:
        ng, wg = r["no_gate"], (r["with_gate"] or {})
        rows.append([r["label"], r["share"], ng.get("macc"), ng.get("sensitivity"),
                     ng.get("specificity"), wg.get("macc"), wg.get("coverage"),
                     ng.get("n")])
    return header, rows, "跨设备域分解"


def table8_audit() -> tuple:
    a = EXP.audit()
    header = ["检查项", "结果", "详情"]
    rows = [[c["name"], "通过" if c["passed"] else "未通过", c["detail"]]
            for c in a["checks"]]
    return header, rows, "数据审计检查"


def table9_stages() -> tuple:
    """质控与分割两个前置阶段的结果。"""
    sq = EXP.sqa_summary()
    sg = EXP.seg_summary()
    header = ["阶段", "指标", "数值", "补充"]
    rows = [
        ["Stage1 质控", "5 折 AUC", sq.get("5fold_mean_auc"),
         "标准差 %.4f" % sq["5fold_std_auc"] if sq.get("5fold_std_auc") else ""],
        ["Stage1 质控", "5 折 Macro-F1", sq.get("5fold_mean_macro_f1"), ""],
        ["Stage1 质控", "跨库零样本 AUC", sq.get("zero_shot_zchsound_auc"),
         "负结果：跨医院分布漂移，支撑域泛化必要性"],
        ["Stage2 分割", "事件级 F1(±60ms)", sg.get("best_event_f1"),
         "最佳 epoch %s" % sg.get("best_epoch")],
        ["Stage2 分割", "最终逐帧准确率", sg.get("final_frame_acc"),
         "训练 %s 轮" % sg.get("epochs")],
    ]
    for k, v in (sq.get("feature_importance") or {}).items():
        rows.append(["Stage1 特征重要性",
                     EXP.C.SQA_FEATURE_LABELS.get(k, k), v, ""])
    return header, rows, "前置两阶段_SQA与分割"


TABLES = [table1_datasets, table23_models, table4_ablation, table5_folds,
          table6_rejection, table7_domain, table8_audit, table9_stages]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print("导出目录: %s\n" % args.out)
    ok = 0
    for fn in TABLES:
        try:
            header, rows, name = fn()
        except Exception as ex:                                  # noqa: BLE001
            print("  FAIL %-24s %s: %s" % (fn.__name__, type(ex).__name__, ex))
            continue
        if not rows:
            print("  SKIP %-24s (无数据，可能 results/ 尚未同步)" % name)
            continue
        path = os.path.join(args.out, name + ".csv")
        n = _write(path, header, rows)
        print("  OK   %-24s %2d 行 x %d 列" % (name, n, len(header)))
        ok += 1

    # 索引
    idx = os.path.join(args.out, "README.md")
    with open(idx, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 竞赛信息表（CSV）\n\n")
        f.write("由 `python tools/export_tables.py` 从 `results/` 真实训练产物自动生成，\n"
                "**不要手工编辑这些 CSV**，改动产物后重跑本脚本即可。\n\n")
        f.write("| 文件 | 内容 |\n|---|---|\n")
        for fn in TABLES:
            try:
                _, _, name = fn()
            except Exception:                                    # noqa: BLE001
                continue
            if os.path.exists(os.path.join(args.out, name + ".csv")):
                f.write("| [%s.csv](%s.csv) | %s |\n" % (name, name, name.replace("_", " ")))
        f.write("\n> 数值口径：MAcc/UAR = (敏感度+特异度)/2，与论文一致；"
                "空单元格表示该产物里没有这个字段，不是 0。\n")
    print("\n索引: docs/tables/README.md")
    print("完成 %d 张表。" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())