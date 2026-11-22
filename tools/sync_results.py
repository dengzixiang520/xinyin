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
    python tools/sync_results.py              # 全量同步 (指标 + 权重 + 图 + 预测)
    python tools/sync_results.py --only-metrics
    python tools/sync_results.py --only-weights
    python tools/sync_results.py --only-predictions
    python tools/sync_results.py --only-checkpoints
    python tools/sync_results.py --dry-run

关于 `results/predictions/`: 这些是**行级**预测 (record_uid + 标签 + 逐条概率),
由医院受控数据派生, 不进版本库, 必须由本工具从训练服务器拉取。
风险-覆盖率 / 校准 / 跨域分解三个页面依赖它; 拉不到时后端返回 available=false,
前端显示 TBD, 不会伪造曲线。
"""
from __future__ import annotations

import argparse
import json
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
PREDDIR = os.path.join(RESULTS, "predictions")

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

# 「多模型对比」页用的对比检查点 (都对应论文里的真实实验)
EXTRA_WEIGHTS = [
    ("paper/outputs/cross_naive_pool_cnn/cls_best.pt", "weights/cls_naive.pt"),
    ("paper/outputs/lodo_proposed_dg/cls_best.pt",     "weights/cls_lodo.pt"),
    ("paper/outputs/abl_no_cwt/cls_best.pt",           "weights/cls_nocwt.pt"),
]

# 论文图
FIGURE_FILES = [
    ("paper/outputs/figures/fig2_tsne_shortcut.png",  "results/figures/fig2_tsne_shortcut.png"),
    ("paper/outputs/figures/fig3_risk_coverage.png",  "results/figures/fig3_risk_coverage.png"),
    ("paper/outputs/figures/fig4_gradcam_case.png",   "results/figures/fig4_gradcam_case.png"),
    ("paper/outputs/figures/fig5_cm_roc.png",         "results/figures/fig5_cm_roc.png"),
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


def write_domain_map() -> bool:
    """从主模型权重里读出 domain_to_idx, 落盘成 results/domain_map.json。

    `/api/selective` 的跨域分解需要 domain_idx -> 域名的映射。原先要手工生成,
    这里做成同步的最后一步, 保证全新克隆 + 一次同步即可完整运行。
    """
    ck = os.path.join(WEIGHTS, "cls_best.pt")
    if not os.path.exists(ck):
        print("  [skip] 无 weights/cls_best.pt, 无法生成 domain_map.json")
        return False
    try:
        import torch
        try:
            state = torch.load(ck, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(ck, map_location="cpu")
        d2i = state.get("domain_to_idx") or {}
        if not d2i:
            print("  [skip] 权重里没有 domain_to_idx")
            return False
        out = {"domain_to_idx": d2i,
               "idx_to_domain": {str(v): k for k, v in d2i.items()},
               "source": "weights/cls_best.pt::domain_to_idx",
               "n_domains": len(d2i)}
        dst = os.path.join(RESULTS, "domain_map.json")
        os.makedirs(RESULTS, exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("  OK   results/domain_map.json (%d 个设备域)" % len(d2i))
        return True
    except Exception as ex:                                      # noqa: BLE001
        print("  [warn] 生成 domain_map.json 失败: %s: %s" % (type(ex).__name__, ex))
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
    ap.add_argument("--only-predictions", action="store_true")
    ap.add_argument("--only-checkpoints", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.host or (not args.password and not args.key):
        print("缺少连接信息。请设置环境变量 SONOTRACE_SSH_HOST / SONOTRACE_SSH_PASS (或 --key)。",
              file=sys.stderr)
        return 2

    c = connect(args)
    sftp = c.open_sftp()
    R = args.remote_root

    selective = (args.only_metrics or args.only_weights
                 or args.only_predictions or args.only_checkpoints)
    do_metrics = args.only_metrics or not selective
    do_weights = args.only_weights or not selective
    do_pred = args.only_predictions or not selective
    do_cks = args.only_checkpoints or not selective

    print("远端根目录: %s" % R)
    print("模式: %s%s%s%s%s" % (
        "dry-run " if args.dry_run else "",
        "metrics " if do_metrics else "",
        "weights " if do_weights else "",
        "predictions " if do_pred else "",
        "checkpoints " if do_cks else ""))

    if do_metrics:
        print("\n[1/5] 固定产物 + 每个实验的 metrics.json")
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

        print("\n[2/5] 论文图")
        for rel, loc in FIGURE_FILES:
            fetch(sftp, posixpath.join(R, rel), os.path.join(LOCAL_ROOT, loc), args.dry_run, loc)

    if do_weights:
        print("\n[3/5] 分类主模型权重 (训练完成后才存在)")
        got = False
        for i, rel in enumerate(CLS_CANDIDATES):
            if fetch(sftp, posixpath.join(R, rel), os.path.join(WEIGHTS, "cls_best.pt"),
                     args.dry_run, "weights/cls_best.pt  <= " + rel):
                got = True
                break
        if not got:
            print("  (cls_best.pt 尚未生成 —— 等 cross_proposed_full 跑完再来同步)")

    if do_cks:
        print("\n[4/5] 多模型对比检查点")
        for rel, loc in EXTRA_WEIGHTS:
            fetch(sftp, posixpath.join(R, rel), os.path.join(LOCAL_ROOT, loc),
                  args.dry_run, loc)

    if do_pred:
        print("\n[5/5] 行级预测 (风险-覆盖率 / 校准 / 跨域分解所需)")
        out_root = posixpath.join(R, METRIC_GLOB_DIR)
        exps = sorted(x for x in remote_list(sftp, out_root) if x not in ("sqa", "seg_unet", "figures"))
        n_ok = 0
        for e in exps:
            rp = posixpath.join(out_root, e, "predictions.csv")
            lp = os.path.join(PREDDIR, e + ".csv")
            if fetch(sftp, rp, lp, args.dry_run, "results/predictions/%s.csv" % e):
                n_ok += 1
        print("  预测文件: %d / %d" % (n_ok, len(exps)))

    if not args.dry_run and (do_weights or do_cks):
        print("\n[+] 由主模型权重反解设备域映射")
        write_domain_map()

    if not args.dry_run and do_metrics:
        # 原始清单含受试者 ID / 医院路径 / 临床分级, 不入库;
        # 后端读的是脱敏版, 所以同步完必须重新生成一次。
        print("\n[+] 生成脱敏清单 (公开仓库只放 public 版)")
        try:
            import subprocess
            subprocess.run([sys.executable,
                            os.path.join(LOCAL_ROOT, "tools", "sanitize_manifest.py")],
                           check=False)
        except Exception as ex:                                  # noqa: BLE001
            print("  [warn] 脱敏清单生成失败: %s: %s" % (type(ex).__name__, ex))
            print("         请手动运行: python tools/sanitize_manifest.py")

    sftp.close()
    c.close()
    print("\n同步完成。重启后端即可加载新权重。")
    return 0


if __name__ == "__main__":
    sys.exit(main())