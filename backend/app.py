#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声见其迹 (SONOTRACE) — 比赛端 FastAPI 后端

启动:
    cd sonotrace-web
    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
浏览器:
    主界面   http://127.0.0.1:8000
    接口文档 http://127.0.0.1:8000/docs

设计原则:
  * 权重缺失时绝不报错, 自动降级为 DSP 声学模式 (答辩现场零翻车)
  * 所有论文指标均从 results/ 真实产物读取, 不硬编码
  * 拒识阈值不是拍脑袋定的, 由 selective.py 在真实外测集上反解
  * 每次分析落库, 拒识记录自动进复核队列
"""
from __future__ import annotations

import hashlib
import os
import time
import tempfile

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import config as C
from . import experiments as EXP
from . import selective as SEL
from . import store as DB
from .engine import REGISTRY, load_audio_from_path, extract_quality
from .pipeline import (analyze, analyze_sample, compare, list_samples,
                       current_gate, GATE_EXPERIMENT)

APP_START = time.time()

app = FastAPI(title=C.APP_TITLE, version=C.APP_VERSION,
              description="生理声音可视化与智能辅助分析系统 — 比赛展示端")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

INDEX = os.path.join(C.FRONTEND_DIR, "index.html")

# 批量分析上限 (一次同步处理太多会长时间阻塞, 超出应改用异步队列)
BATCH_MAX = 12
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


@app.on_event("startup")
def _startup():
    DB.init_db()


# ==================================================================== 前端托管
@app.get("/", include_in_schema=False)
def index():
    if not os.path.exists(INDEX):
        return JSONResponse({"detail": "frontend/index.html 不存在"}, status_code=404)
    return FileResponse(INDEX, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    p = os.path.join(C.FRONTEND_DIR, "favicon.ico")
    if os.path.exists(p):
        return FileResponse(p)
    return JSONResponse({}, status_code=204)


# ==================================================================== 健康 / 模型
@app.get("/api/health", summary="服务与权重健康检查")
def health():
    st = REGISTRY.status()
    return {"status": "ok", "system": C.APP_TITLE, "version": C.APP_VERSION,
            "uptime_sec": round(time.time() - APP_START, 1),
            "engine_mode": st["engine_mode"], "device": st["device"],
            "weights": st["weights"], "cls": st["cls"], "seg": st["seg"],
            "sqa": st["sqa"], "n_checkpoints": st["n_available"]}


@app.get("/api/version", summary="版本与运行环境")
def version():
    import platform
    import numpy
    try:
        import torch
        tv = torch.__version__
        cuda = torch.cuda.is_available()
        gpu = (torch.cuda.get_device_name(0) if cuda else None)
    except Exception:                                            # noqa: BLE001
        tv, cuda, gpu = None, False, None
    return {"app": C.APP_TITLE, "version": C.APP_VERSION,
            "python": platform.python_version(), "numpy": numpy.__version__,
            "torch": tv, "cuda_available": cuda, "gpu": gpu,
            "uptime_sec": round(time.time() - APP_START, 1)}


@app.get("/api/models", summary="三项模型的加载详情")
def models():
    return REGISTRY.status()


@app.get("/api/checkpoints", summary="多模型对比池 (同一片段可换模型重算)")
def checkpoints():
    st = REGISTRY.status()
    return {"default": C.DEFAULT_CHECKPOINT,
            "items": st["checkpoints"], "n_available": st["n_available"]}


# ==================================================================== 论文实验（真实指标）
@app.get("/api/overview", summary="首页仪表盘聚合数字")
def overview():
    d = EXP.overview()
    d["gate"] = current_gate(None)
    d["review"] = DB.stats()
    return d


@app.get("/api/experiments", summary="全部论文实验真实指标 (Table 2/3/4/5)")
def experiments():
    return EXP.experiments()


@app.get("/api/experiments/{name}", summary="单个实验详情")
def experiment_detail(name: str):
    data = EXP.experiments()
    for it in data["items"]:
        if it["name"] == name:
            return it
    raise HTTPException(status_code=404, detail=f"unknown experiment: {name}")


@app.get("/api/sqa", summary="Stage1 六维 SQA 质控标定结果")
def sqa():
    return EXP.sqa_summary()


@app.get("/api/seg", summary="Stage2 1D U-Net 分割训练曲线")
def seg():
    return EXP.seg_summary()


@app.get("/api/datasets", summary="Table1 数据集统计")
def datasets():
    return EXP.dataset_table()


@app.get("/api/ablation", summary="Table4 消融差值表 (相对完整模型)")
def ablation():
    return EXP.ablation_table()


@app.get("/api/audit", summary="数据可追溯性与隔离性检查")
def audit():
    return EXP.audit()


# ==================================================================== 风险-覆盖率
@app.get("/api/selective", summary="风险-覆盖率 / 校准 / 跨域分解 (完整)")
def selective(experiment: str = Query(GATE_EXPERIMENT,
                                      description="实验名, 见 /api/selective/experiments")):
    return SEL.summary(experiment)


@app.get("/api/selective/experiments", summary="可用于风险-覆盖率分析的实验清单")
def selective_experiments():
    return {"experiments": SEL.available(), "default": GATE_EXPERIMENT}


@app.get("/api/selective/{name}", summary="指定实验的风险-覆盖率分析")
def selective_one(name: str):
    d = SEL.summary(name)
    if not d.get("available"):
        raise HTTPException(status_code=404,
                            detail=f"no predictions for experiment: {name}")
    return d


@app.get("/api/gate", summary="当前生效的拒识门控参数与三个档位")
def gate():
    return {"current": current_gate(None),
            "calibrated_on": GATE_EXPERIMENT,
            "modes": SEL.modes(GATE_EXPERIMENT),
            "explain": ("门 1 认知不确定度: MC-Dropout 反复采样仍不一致 -> 拒识; "
                        "门 2 决策边际: 概率贴近 0.5 等价于掷硬币 -> 拒识。"
                        "阈值按目标拒识率在真实外测集上反解, 档位间覆盖率与准确率的"
                        "测算值均由 results/predictions/ 逐条重算。")}


# ==================================================================== 样本与分析
@app.get("/api/samples", summary="内置演示心音样本列表")
def samples():
    return {"samples": list_samples()}


@app.get("/api/sample_audio/{name}", summary="回放内置样本原始 .wav (供前端 Web Audio 解码)")
def sample_audio(name: str):
    safe = os.path.basename(name)                 # 防目录穿越
    p = os.path.join(C.SAMPLES_DIR, safe)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail=f"sample not found: {safe}")
    return FileResponse(p, media_type="audio/wav", filename=safe)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _persist(result: dict, sha: str | None) -> dict:
    """落库 + 回填 id; 数据库不可用时不影响分析结果本身。"""
    try:
        aid = DB.save_analysis(result, audio_sha256=sha)
        result["analysis_id"] = aid
    except Exception:                                            # noqa: BLE001
        result["analysis_id"] = None
    return result


@app.post("/api/analyze_sample", summary="分析内置样本 (无需上传)")
def analyze_builtin(name: str = Query(..., description="samples/ 下的文件名"),
                    gradcam: bool = Query(True),
                    checkpoint: str | None = Query(None),
                    gate_mode: str | None = Query(None),
                    persist: bool = Query(True, description="是否写入分析历史与复核队列")):
    try:
        r = analyze_sample(name, include_gradcam=gradcam,
                           checkpoint=checkpoint, gate_mode=gate_mode)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"sample not found: {name}")
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")
    # 前端每次打开页面会自动分析一次首样本以填充仪表盘 —— 那是展示动作,
    # 不该污染复核队列。页面加载路径传 persist=false, 用户主动分析才落库。
    return _persist(r, None) if persist else {**r, "analysis_id": None, "persisted": False}


@app.post("/api/analyze", summary="上传 .wav 心音文件并完整分析")
async def analyze_upload(file: UploadFile = File(...),
                         gradcam: bool = Query(True),
                         checkpoint: str | None = Query(None),
                         gate_mode: str | None = Query(None),
                         persist: bool = Query(True)):
    raw = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        r = analyze(tmp_path, filename=file.filename or "upload.wav",
                    include_gradcam=gradcam, checkpoint=checkpoint,
                    gate_mode=gate_mode)
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return _persist(r, _sha256_bytes(raw)) if persist else {**r, "analysis_id": None,
                                                           "persisted": False}


@app.post("/api/analyze_batch", summary="批量分析多个心音文件 (同步, 上限 12 个)")
async def analyze_batch(files: list[UploadFile] = File(...),
                        gradcam: bool = Query(False),
                        checkpoint: str | None = Query(None),
                        gate_mode: str | None = Query(None)):
    if len(files) > BATCH_MAX:
        raise HTTPException(status_code=400,
                            detail=f"一次最多 {BATCH_MAX} 个文件, 收到 {len(files)} 个")
    gate = current_gate(gate_mode)
    rows, errors = [], []
    t0 = time.time()
    for uf in files:
        nm = uf.filename or "unnamed.wav"
        if not nm.lower().endswith(AUDIO_EXT):
            errors.append({"filename": nm, "error": "不支持的音频扩展名"})
            continue
        raw = await uf.read()
        tmp_path = None
        try:
            suffix = os.path.splitext(nm)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            r = analyze(tmp_path, filename=nm, include_gradcam=gradcam,
                        checkpoint=checkpoint, gate_mode=gate_mode)
            pr = r["prediction"]
            rows.append({
                "filename": nm, "duration_sec": r["duration_sec"],
                "quality_passed": r["quality"]["passed"],
                "prob_abnormal": pr.get("prob_abnormal"),
                "uncertainty": pr.get("uncertainty"),
                "verdict": pr.get("label"),
                "verdict_key": pr.get("verdict_key"),
                "gate_hit": pr.get("gate_hit"),
                "analysis_id": _persist(r, _sha256_bytes(raw))["analysis_id"],
            })
        except Exception as ex:                                  # noqa: BLE001
            errors.append({"filename": nm, "error": f"{type(ex).__name__}: {ex}"})
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    ok = [r for r in rows if r["verdict_key"] == "abnormal"]
    nor = [r for r in rows if r["verdict_key"] == "normal"]
    pen = [r for r in rows if r["verdict_key"] == "pending"]
    return {
        "gate": gate,
        "n_total": len(files), "n_ok": len(rows), "n_error": len(errors),
        "elapsed_sec": round(time.time() - t0, 2),
        "summary": {"abnormal": len(ok), "normal": len(nor),
                    "pending": len(pen),
                    "abstain_rate": round(len(pen) / len(rows), 4) if rows else None},
        "items": rows, "errors": errors,
        "note": ("verdict_key=pending 表示「质控未过」或「门控拒识」, 需人工复核; "
                 "批量结果已落库, 可在复核队列中处理。"),
    }


@app.post("/api/compare", summary="同一片段在多个检查点上并排推理 (上传)")
async def compare_upload(file: UploadFile = File(...),
                         checkpoints: str | None = Query(
                             None, description="逗号分隔的检查点名, 缺省全部")):
    raw = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    want = [s.strip() for s in checkpoints.split(",") if s.strip()] if checkpoints else None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        return compare(tmp_path, filename=file.filename or "upload.wav",
                       checkpoints=want)
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/compare_sample", summary="同一内置样本在多个检查点上并排推理")
def compare_sample(name: str = Query(...),
                   checkpoints: str | None = Query(None)):
    p = os.path.join(C.SAMPLES_DIR, os.path.basename(name))
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail=f"sample not found: {name}")
    want = [s.strip() for s in checkpoints.split(",") if s.strip()] if checkpoints else None
    try:
        return compare(p, filename=name, checkpoints=want)
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")


@app.post("/api/quality_check", summary="仅执行六维前置质控 (ESP32 硬件可先调此接口拦截)")
async def quality_check(file: UploadFile = File(...)):
    raw = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        _, raw_y, sr = load_audio_from_path(tmp_path)
        return extract_quality(raw_y, sr)
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================================================================== 历史 / 复核队列
@app.get("/api/history", summary="分析历史 (不含完整 payload)")
def history(limit: int = Query(50, ge=1, le=500),
            offset: int = Query(0, ge=0),
            verdict_key: str | None = Query(None)):
    return DB.list_analyses(limit=limit, offset=offset, verdict_key=verdict_key)


@app.get("/api/history/{analysis_id}", summary="单条分析详情 (含完整 payload)")
def history_detail(analysis_id: int):
    d = DB.get_analysis(analysis_id)
    if d is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return d


@app.get("/api/review-queue", summary="待人工复核队列")
def review_queue(status: str = Query("open"), limit: int = Query(50, ge=1, le=500),
                 offset: int = Query(0, ge=0)):
    return DB.list_reviews(status=status, limit=limit, offset=offset)


@app.post("/api/review-queue/{review_id}", summary="复核结论 (确认/否定/升级)")
def review_resolve(review_id: int,
                   status: str = Body(..., embed=True,
                                      description="confirmed | rejected | escalated"),
                   reviewer: str = Body("clinician", embed=True),
                   note: str = Body("", embed=True)):
    try:
        ok = DB.resolve_review(review_id, status, reviewer=reviewer, note=note)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    if not ok:
        raise HTTPException(status_code=404, detail="review not found")
    return {"ok": True, "review_id": review_id, "status": status,
            "stats": DB.stats()}


@app.get("/api/stats", summary="分析记录与复核统计")
def stats():
    return DB.stats()


@app.post("/api/maintenance/prune", summary="清理超过 N 天的分析记录")
def prune(keep_days: int = Query(30, ge=1, le=3650)):
    return {"deleted": DB.prune(keep_days=keep_days), "keep_days": keep_days}