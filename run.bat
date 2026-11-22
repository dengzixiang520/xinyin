@echo off
REM 声见其迹 SONOTRACE 一键启动 (Windows)
cd /d "%~dp0"
if not exist venv (
  python -m venv venv
  venv\Scripts\pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)
venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
