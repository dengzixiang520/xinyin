#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声见其迹 SONOTRACE — 论文产物同步工具
把 3090 服务器上 paper/outputs 的真实训练产物同步到比赛端 results/ 与 weights/。

凭据一律从环境变量读取, **不写入仓库**:
    SONOTRACE_SSH_HOST   例 1.2.3.4
    SONOTRACE_SSH_PORT   例 33668
    SONOTRACE_SSH_USER   例 linux
    SONOTRACE_SSH_PASS   例 ******
    SONOTRACE_SSH_KEY    可选, 私钥路径 (优先于密码)
    SONOTRACE_REMOTE_ROOT 可选, 默认 /home/linux/sonotrace

用法:
    python tools/sync_results.py              # 全量同步 (指标 + 权重 + 图)
    python tools/sync_results.py --only-metrics
    python tools/sync_results.py --only-weights
    python tools/sync_results.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import posixpath
import sys

try:
    import paramiko
except ImportError:
    print("需要 paramiko:  pip install paramiko", file=sys.stderr)
    sys.exit(2)

LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(LOCAL_ROOT, "results")
EXPDIR = os.path.join(RESULTS, "experiments")
WEIGHTS = os.path.join(LOCAL_ROOT, "weights")
FIGDIR = os.path.join(RESULTS, "figures")

# 远端 -> 本地 的固定映射
SINGLE_FILES = [
    ("paper/outputs/sqa/sqa_results.json",            "results/sqa_results.json"),
    ("paper/outputs/sqa/sqa_6d_features.csv",         "results/sqa_6d_features.csv"),
    ("paper/outputs/sqa/sqa_rf_model.pkl",            "weights/sqa_rf_model.pkl"),
    ("paper/outputs/seg_unet/seg_history_fold0.json", "results/seg_history_fold0.json"),
    ("paper/outputs/seg_unet/seg_unet_fold0.pt",      "weights/seg_unet.pt"),
    ("paper/manifests/data_audit_report.json",        "results/data_audit_report.json"),
    ("paper/manifests/master_pcg_manifest.csv",       "results/master_pcg_manifest.csv"),
]

# 需要按目录扫描的: outputs/<exp>/metrics.json -> results/experiments/<exp>.json
METRIC_GLOB_DIR = "paper/outputs"

# 训练完成后才应导出的主模型权重
CLS_CANDIDATES = [
    "paper/outputs/cross_proposed_full/cls_best.pt",
    "paper/outputs/lodo_proposed_dg/cls_best.pt",
]

# 论文图
FIGURE_FILES = [
    ("paper/outputs/figures/fig2_tsne_shortcut.png", "results/figures/fig2_tsne_shortcut.png"),
    ("paper/outputs/figures/fig3_risk_coverage.png", "results/figures/fig3_risk_coverage.png"),
    ("paper/outputs/figures/fig5_cm_roc.png",        "results/figures/fig5_cm_roc.png"),
]


def connect(args):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(port=args.port, username=args.user, timeout=20,
              banner_timeout=30, auth_timeout=30,
              allow_agent=False, look_for_keys=False)
    if args.key:
        c.connect(args.host, key_filename=args.key, **kw)
    else:
        c.connect(args.host, password=args.password, **kw)
    return c


def remote_list(sftp, path):
    try:
        return sftp.listdir(path)
    except IOError:
        return []


def fetch(sftp, remote, local, dry, tag=""):
    if dry:
        try:
            sz = sftp.stat(remote).st_size
        except IOError:
            print("  [dry] MISSING %s" % tag); return False
        print("  [dry] %-46s %9.1f KB" % (tag, sz / 1024))
        return True
    os.makedirs(os.path.dirname(local), exist_ok=True)
    try:
        sftp.get(remote, local)
        # 拒绝 0 字节 / 空内容 (远端文件可能正在写入)
        if os.path.getsize(local) == 0:
            os.remove(local)
            print("  EMPT %-46s (远端文件为空, 已跳过)" % tag)
            return False
        print("  OK   %-46s %9.1f KB" % (tag, os.path.getsize(local) / 1024))
        return True
    except IOError as ex:
        # 清理可能被部分写入的本地文件, 避免污染 results/
        if os.path.exists(local) and os.path.getsize(local) == 0:
            os.remove(local)
        print("  MISS %-46s %s" % (tag, ex))
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("SONOTRACE_SSH_HOST"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SONOTRACE_SSH_PORT", "22")))
    ap.add_argument("--user", default=os.environ.get("SONOTRACE_SSH_USER", "linux"))
    ap.add_argument("--password", default=os.environ.get("SONOTRACE_SSH_PASS"))
    ap.add_argument("--key", default=os.environ.get("SONOTRACE_SSH_KEY"))
    ap.add_argument("--remote-root", default=os.environ.get("SONOTRACE_REMOTE_ROOT", "/home/linux/sonotrace"))
    ap.add_argument("--only-metrics", action="store_true")
    ap.add_argument("--only-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.host or (not args.password and not args.key):
        print("缺少连接信息。请设置环境变量 SONOTRACE_SSH_HOST / SONOTRACE_SSH_PASS (或 --key)。",
              file=sys.stderr)
        return 2

    c = connect(args)
    sftp = c.open_sftp()
    R = args.remote_root

    do_metrics = not args.only_weights
    do_weights = not args.only_metrics

    print("远端根目录: %s" % R)
    print("模式: %s%s%s" % (
        "dry-run " if args.dry_run else "",
        "metrics " if do_metrics else "",
        "weights " if do_weights else ""))

    if do_metrics:
        print("\n[1/3] 固定产物 + 每个实验的 metrics.json")
        for rel, loc in SINGLE_FILES:
            if not do_weights and loc.startswith("weights/"):
                continue
            fetch(sftp, posixpath.join(R, rel), os.path.join(LOCAL_ROOT, loc),
                  args.dry_run, loc)

        out_root = posixpath.join(R, METRIC_GLOB_DIR)
        exps = sorted(x for x in remote_list(sftp, out_root) if x not in ("sqa", "seg_unet", "figures"))
        n_ok = 0
        for e in exps:
            rp = posixpath.join(out_root, e, "metrics.json")
            lp = os.path.join(EXPDIR, e + ".json")
            if fetch(sftp, rp, lp, args.dry_run, "results/experiments/%s.json" % e):
                n_ok += 1
        print("  已完成实验指标: %d / %d" % (n_ok, len(exps)))

        print("\n[2/3] 论文图")
        for rel, loc in FIGURE_FILES:
            fetch(sftp, posixpath.join(R, rel), os.path.join(LOCAL_ROOT, loc), args.dry_run, loc)

    if do_weights:
        print("\n[3/3] 分类主模型权重 (训练完成后才存在)")
        got = False
        for i, rel in enumerate(CLS_CANDIDATES):
            if fetch(sftp, posixpath.join(R, rel), os.path.join(WEIGHTS, "cls_best.pt"),
                     args.dry_run, "weights/cls_best.pt  <= " + rel):
                got = True
                break
        if not got:
            print("  (cls_best.pt 尚未生成 —— 等 cross_proposed_full 跑完再来同步)")

    sftp.close()
    c.close()
    print("\n同步完成。重启后端即可加载新权重。")
    return 0


if __name__ == "__main__":
    sys.exit(main())