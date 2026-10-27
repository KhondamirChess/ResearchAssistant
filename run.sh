#!/usr/bin/env bash
# One-command start on Ubuntu:  ./run.sh   then open http://localhost:8000
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
exec uvicorn backend.app:app --reload --port 8000
