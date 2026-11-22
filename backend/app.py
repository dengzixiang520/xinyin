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
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import config as C
from . import experiments as EXP
from .engine import REGISTRY, load_audio_from_path, extract_quality
from .pipeline import analyze, analyze_sample, list_samples

app = FastAPI(title=C.APP_TITLE, version=C.APP_VERSION,
              description="生理声音可视化与智能辅助分析系统 — 比赛展示端")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

INDEX = os.path.join(C.FRONTEND_DIR, "index.html")


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
            "engine_mode": st["engine_mode"], "device": st["device"],
            "weights": st["weights"], "cls": st["cls"], "seg": st["seg"], "sqa": st["sqa"]}


@app.get("/api/models", summary="三项模型的加载详情")
def models():
    return REGISTRY.status()


# ==================================================================== 论文实验（真实指标）
@app.get("/api/overview", summary="首页仪表盘聚合数字")
def overview():
    return EXP.overview()


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


@app.post("/api/analyze_sample", summary="分析内置样本 (无需上传)")
def analyze_builtin(name: str = Query(..., description="samples/ 下的文件名"),
                    gradcam: bool = Query(True)):
    try:
        return analyze_sample(name, include_gradcam=gradcam)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"sample not found: {name}")
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")


@app.post("/api/analyze", summary="上传 .wav 心音文件并完整分析")
async def analyze_upload(file: UploadFile = File(...),
                         gradcam: bool = Query(True)):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        return analyze(tmp_path, filename=file.filename or "upload.wav",
                       include_gradcam=gradcam)
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/quality_check", summary="仅执行六维前置质控 (ESP32 硬件可先调此接口拦截)")
async def quality_check(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        _, raw_y, sr = load_audio_from_path(tmp_path)
        return extract_quality(raw_y, sr)
    except Exception as ex:                                      # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(ex).__name__}: {ex}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)