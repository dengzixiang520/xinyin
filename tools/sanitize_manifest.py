#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声见其迹 SONOTRACE — 清单脱敏工具

`master_pcg_manifest.csv` 是**行级临床表**，含受试者 ID、年龄段、采集站点、
临床分级、医院内部相对路径。它来自受控访问数据集（CirCor / PhysioNet）
与医院合作数据（ZCHSound），**不适合公开到 GitHub**。

本脚本生成一份**脱敏但仍有核验价值**的公开清单：
  * 去掉  record_uid / rel_path / age_group / site / label_fine
          / label_binary_clinical / site_has_murmur / sqi_label / seg_tsv_rel_path
  * subject_id  →  subject_hash = sha256(dataset|subject_id) 的前 12 位
    （同一受试者哈希相同，因此**外测集隔离**与**受试者级折内唯一**两项检查仍然成立；
      但无法由哈希反推原始 ID）
  * rel_path    →  path_present (0/1)
    （保留「路径字段完整性」这项检查的含义，但不泄露任何真实路径）
  * 其余统计所需列原样保留：dataset / domain_id / duration_sec / sr / channels
    / label_binary_acoustic / fold_internal / has_seg_tsv

用法:
    python tools/sanitize_manifest.py            # 生成 public 版本
    python tools/sanitize_manifest.py --check    # 只核对，不写文件

脱敏后 `results/master_pcg_manifest.public.csv` 才是入库文件；
原始 `master_pcg_manifest.csv` 已被 .gitignore 排除，只留在本地。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("需要 pandas", file=sys.stderr)
    sys.exit(2)

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(LOCAL_ROOT, "results", "master_pcg_manifest.csv")
PUB = os.path.join(LOCAL_ROOT, "results", "master_pcg_manifest.public.csv")

# 明确要丢掉的敏感列
DROP = ["record_uid", "rel_path", "age_group", "site", "label_fine",
        "label_binary_clinical", "site_has_murmur", "sqi_label",
        "seg_tsv_rel_path"]

# 公开版本保留的列（顺序即输出顺序）
KEEP = ["dataset", "subject_hash", "domain_id", "duration_sec", "sr", "channels",
        "label_binary_acoustic", "fold_internal", "has_seg_tsv", "path_present"]


def subject_hash(dataset: str, subject: str) -> str:
    """确定性哈希：同一 (数据集, 受试者) 恒定得到同一值。

    加盐固定值只用于去关联，不是安全措施；目的是让公开文件里不出现原始 ID。
    """
    token = "sonotrace:%s|%s" % (str(dataset), str(subject))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只核对不写文件")
    args = ap.parse_args()

    if not os.path.exists(RAW):
        print("找不到原始清单: %s" % RAW, file=sys.stderr)
        print("先运行 tools/sync_results.py 从训练服务器同步。", file=sys.stderr)
        return 2

    df = pd.read_csv(RAW)
    print("原始清单: %d 行 x %d 列" % (len(df), len(df.columns)))

    missing = [c for c in ("dataset", "subject_id", "duration_sec") if c not in df.columns]
    if missing:
        print("原始清单缺少必需列: %s" % missing, file=sys.stderr)
        return 2

    out = pd.DataFrame()
    out["dataset"] = df["dataset"]
    # 用向量化的方式算哈希（11k 行，直接 map 即可）
    pairs = list(zip(df["dataset"].astype(str), df["subject_id"].astype(str)))
    out["subject_hash"] = [subject_hash(d, s) for d, s in pairs]
    out["domain_id"] = df.get("domain_id")
    out["duration_sec"] = df.get("duration_sec")
    out["sr"] = df.get("sr")
    out["channels"] = df.get("channels")
    out["label_binary_acoustic"] = df.get("label_binary_acoustic")
    out["fold_internal"] = df.get("fold_internal")
    out["has_seg_tsv"] = df.get("has_seg_tsv")
    out["path_present"] = (df["rel_path"].notna().astype(int)
                           if "rel_path" in df.columns else 1)
    out = out[KEEP]

    # ---- 核验：脱敏不能破坏可验证性 ----
    ok = True
    n_raw_subj = df.groupby("dataset")["subject_id"].nunique().sum()
    n_pub_subj = out.groupby("dataset")["subject_hash"].nunique().sum()
    print("\n=== 脱敏后核验 ===")
    print("  行数        %d -> %d  %s" % (len(df), len(out), "OK" if len(df) == len(out) else "MISMATCH"))
    ok &= len(df) == len(out)
    print("  受试者数    %d -> %d  %s" % (n_raw_subj, n_pub_subj,
                                          "OK" if n_raw_subj == n_pub_subj else "MISMATCH"))
    ok &= n_raw_subj == n_pub_subj

    # 外测集隔离检查在脱敏后必须仍能判定
    train = set(out[out["dataset"].isin(["PhysioNet_2016", "CirCor_2022",
                                         "BMD_HS_2024", "Fraiwan_2025"])]["subject_hash"])
    test = set(out[out["dataset"] == "ZCHSound_2024"]["subject_hash"])
    print("  外测集交集  %d 人  %s" % (len(train & test),
                                       "OK" if not (train & test) else "LEAK!"))
    ok &= not (train & test)

    # 受试者级折内唯一
    asg = out[out["fold_internal"].notna() & (out["fold_internal"] >= 0)]
    g = asg.groupby(["dataset", "subject_hash"])["fold_internal"].nunique()
    print("  跨折受试者  %d 人  %s" % (int((g > 1).sum()), "OK" if not (g > 1).any() else "MISMATCH"))
    ok &= not (g > 1).any()

    leaked_cols = [c for c in DROP if c in out.columns]
    print("  敏感列残留  %s" % (leaked_cols or "无"))
    ok &= not leaked_cols

    if args.check:
        print("\n[check] 仅核对，未写文件。结论: %s" % ("通过" if ok else "不通过"))
        return 0 if ok else 1

    if not ok:
        print("\n核验未通过，拒绝写出脱敏文件。", file=sys.stderr)
        return 1

    out.to_csv(PUB, index=False, encoding="utf-8")
    print("\n已写出: results/master_pcg_manifest.public.csv (%.1f KB)"
          % (os.path.getsize(PUB) / 1024))
    print("原始清单保留在本地，已由 .gitignore 排除，不入库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())