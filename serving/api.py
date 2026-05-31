"""
serving/api.py
───────────────
FastAPI endpoint that serves real-time recommendations.

Usage:
    uvicorn serving.api:app --reload --port 8000

Endpoints:
    GET  /recommend/{user_idx}   → top-N item indices + latency
    GET  /health                 → service health check
    POST /embed/user             → get raw user embedding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import json
import pickle
import numpy as np
import pandas as pd
import faiss
import lightgbm as lgb
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import PROCESSED_DIR, ARTIFACT_DIR, RETRIEVAL_TOP_K, RANKER_TOP_N
from retrieval.faiss_index import load_model, retrieve
from ranking.features import build_user_stats, build_item_stats, build_ranking_features
from utils.logger import get_logger

log = get_logger(__name__)
app = FastAPI(title="H&M RecSys API", version="1.0")


# ── Startup: load all artifacts once ──────────────────────────────────────

class ModelStore:
    def __init__(self):
        log.info("Loading artifacts …")

        with open(ARTIFACT_DIR / "model_meta.json") as f:
            self.meta = json.load(f)
        with open(ARTIFACT_DIR / "ranker_meta.json") as f:
            ranker_meta = json.load(f)
        with open(PROCESSED_DIR / "train_interactions.pkl", "rb") as f:
            self.train_interactions = pickle.load(f)

        self.user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
        self.item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")
        train_tx = pd.read_parquet(PROCESSED_DIR / "train.parquet")

        self.model  = load_model(self.meta, self.user_features, self.item_features)
        self.index  = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
        self.index.nprobe = 10
        self.item_embeddings = np.load(ARTIFACT_DIR / "item_embeddings.npy")
        self.ranker = lgb.Booster(model_file=str(ARTIFACT_DIR / "ranker.lgb"))
        self.feature_cols = ranker_meta["feature_cols"]

        self.user_stats = build_user_stats(train_tx)
        self.item_stats  = build_item_stats(train_tx)

        log.info("All artifacts loaded. API ready.")


store = ModelStore()


# ── Request / Response schemas ─────────────────────────────────────────────

class RecommendResponse(BaseModel):
    user_idx: int
    recommendations: list[int]
    retrieval_latency_ms: float
    ranking_latency_ms: float
    total_latency_ms: float


class EmbedRequest(BaseModel):
    user_idx: int


class EmbedResponse(BaseModel):
    user_idx: int
    embedding: list[float]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_idx}", response_model=RecommendResponse)
def recommend(user_idx: int, top_n: int = RANKER_TOP_N):
    uf = store.user_features.set_index("user_idx")
    if user_idx not in uf.index:
        raise HTTPException(status_code=404, detail=f"user_idx {user_idx} not found")

    # ── Stage 1: Retrieval ──
    t0 = time.perf_counter()
    seen       = set(store.train_interactions.get(user_idx, []))
    candidates = retrieve(store.model, user_idx, store.user_features,
                          store.meta["user_feat_cols"], store.index,
                          top_k=RETRIEVAL_TOP_K, exclude_seen=seen)
    t1 = time.perf_counter()
    retrieval_ms = (t1 - t0) * 1000

    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates found")

    # ── Stage 2: Ranking ──
    row = uf.loc[user_idx]
    device_t = next(store.model.parameters()).device
    uid_t = torch.tensor([user_idx], device=device_t)
    uf_t  = {c: torch.tensor([int(row[c])], device=device_t)
             for c in store.meta["user_feat_cols"]}

    with torch.no_grad():
        u_emb = store.model.get_user_embedding(uid_t, uf_t).cpu().numpy().astype("float32")

    item_emb_matrix = store.item_embeddings[candidates]
    scores = (u_emb @ item_emb_matrix.T).squeeze()
    if scores.ndim == 0:
        scores = np.array([float(scores)])

    pairs = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates,
                           "retrieval_score": scores.tolist(), "label": 0})
    feat_df, _ = build_ranking_features(pairs, store.user_stats, store.item_stats,
                                         store.item_features, store.user_features)
    X = feat_df[store.feature_cols].values
    pred_scores  = store.ranker.predict(X)
    ranked = [candidates[i] for i in np.argsort(-pred_scores)]
    t2 = time.perf_counter()
    ranking_ms = (t2 - t1) * 1000

    return RecommendResponse(
        user_idx=user_idx,
        recommendations=ranked[:top_n],
        retrieval_latency_ms=round(retrieval_ms, 2),
        ranking_latency_ms=round(ranking_ms, 2),
        total_latency_ms=round(retrieval_ms + ranking_ms, 2),
    )


@app.post("/embed/user", response_model=EmbedResponse)
def embed_user(req: EmbedRequest):
    uf = store.user_features.set_index("user_idx")
    if req.user_idx not in uf.index:
        raise HTTPException(status_code=404, detail=f"user_idx {req.user_idx} not found")

    row = uf.loc[req.user_idx]
    device_t = next(store.model.parameters()).device
    uid_t = torch.tensor([req.user_idx], device=device_t)
    uf_t  = {c: torch.tensor([int(row[c])], device=device_t)
             for c in store.meta["user_feat_cols"]}

    with torch.no_grad():
        emb = store.model.get_user_embedding(uid_t, uf_t).cpu().numpy().squeeze().tolist()

    return EmbedResponse(user_idx=req.user_idx, embedding=emb)
