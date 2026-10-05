#!/usr/bin/env bash
# Convenience script: create venv, install pinned deps, start the API.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3.12 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Created .env - edit it and set GEMINI_API_KEY, then re-run ./run.sh"
  exit 0
fi

uvicorn app.main:app --reload --port 8000
