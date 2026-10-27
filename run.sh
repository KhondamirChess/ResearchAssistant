#!/usr/bin/env bash
# One-command start on Ubuntu:  ./run.sh   then open http://localhost:8000
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
# вызываем pip и uvicorn через python из .venv: так работает, даже если папку проекта перенесли
.venv/bin/python -m pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
exec .venv/bin/python -m uvicorn backend.app:app --reload --port 8000
