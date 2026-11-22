#!/usr/bin/env bash
# 声见其迹 SONOTRACE 一键启动 (Linux/macOS)
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
exec ./venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
