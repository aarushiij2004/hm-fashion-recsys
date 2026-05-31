#!/bin/bash
set -e

echo "=== H&M RecSys Pipeline ==="

echo "[1/5] Preprocessing data …"
python data/preprocess.py

echo "[2/5] Training two-tower model …"
python retrieval/train.py

echo "[3/5] Building FAISS index …"
python retrieval/faiss_index.py --mode build

echo "[4/5] Training LightGBM ranker …"
python ranking/ranker.py --mode train

echo "[5/5] Starting API server …"
uvicorn serving.api:app --host 0.0.0.0 --port 8000

echo "=== Done. API running at http://localhost:8000 ==="
