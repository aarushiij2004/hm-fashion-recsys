"""
ranking/ranker.py
──────────────────
Train a LightGBM LambdaRank model on (user, candidate_item) pairs.
Positive label = item the user actually purchased in the val window.

Usage:
    python ranking/ranker.py --mode train
    python ranking/ranker.py --mode infer --user_idx 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import faiss

from config import (PROCESSED_DIR, ARTIFACT_DIR, RETRIEVAL_TOP_K,
                    RANKER_TOP_N, LGBM_PARAMS, DEVICE)
from retrieval.faiss_index import load_model, retrieve
from ranking.features import build_user_stats, build_item_stats, build_ranking_features
from utils.metrics import ndcg_at_k, recall_at_k
from utils.logger import get_logger

log = get_logger(__name__)


# ── Build training data for ranker ─────────────────────────────────────────

def build_train_pairs(val_interactions: dict, train_interactions: dict,
                      index: faiss.Index, model, user_features: pd.DataFrame,
                      item_features: pd.DataFrame, meta: dict,
                      n_users: int = 5000) -> pd.DataFrame:
    """
    For a sample of users, retrieve candidates from FAISS,
    label positives (purchased in val), negatives (not purchased).
    """
    log.info("Building ranker training pairs …")
    user_feat_cols = meta["user_feat_cols"]
    val_users = list(val_interactions.keys())

    import random
    sample_users = random.sample(val_users, min(n_users, len(val_users)))

    rows = []
    for user_idx in sample_users:
        seen  = set(train_interactions.get(user_idx, []))
        positives = set(val_interactions.get(user_idx, []))
        if not positives:
            continue

        candidates = retrieve(model, user_idx, user_features, user_feat_cols,
                              index, top_k=RETRIEVAL_TOP_K, exclude_seen=seen)
        if not candidates:
            continue

        # Get retrieval scores (dot product)
        import torch
        from config import EMBEDDING_DIM
        uf = user_features.set_index("user_idx")
        if user_idx not in uf.index:
            continue
        row = uf.loc[user_idx]
        import torch
        device_t = next(model.parameters()).device
        uid_t = torch.tensor([user_idx], device=device_t)
        uf_t  = {c: torch.tensor([int(row[c])], device=device_t) for c in user_feat_cols}
        with torch.no_grad():
            u_emb = model.get_user_embedding(uid_t, uf_t).cpu().numpy().astype("float32")

        item_emb_matrix = np.load(ARTIFACT_DIR / "item_embeddings.npy")[candidates]
        scores = (u_emb @ item_emb_matrix.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        for cand_item, score in zip(candidates, scores):
            rows.append({
                "user_idx": user_idx,
                "item_idx": cand_item,
                "retrieval_score": float(score),
                "label": int(cand_item in positives),
            })

    df = pd.DataFrame(rows)
    log.info(f"Ranker pairs: {len(df):,}  positives: {df['label'].sum():,}")
    return df


# ── Train ──────────────────────────────────────────────────────────────────

def train():
    log.info("Loading artifacts …")
    with open(ARTIFACT_DIR / "model_meta.json") as f:
        meta = json.load(f)

    user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")
    train_tx = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    with open(PROCESSED_DIR / "train_interactions.pkl", "rb") as f:
        train_interactions = pickle.load(f)
    with open(PROCESSED_DIR / "val_interactions.pkl", "rb") as f:
        val_interactions = pickle.load(f)

    model = load_model(meta, user_features, item_features)
    index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
    index.nprobe = 10

    # Build train pairs (retrieval → label)
    pairs_df = build_train_pairs(val_interactions, train_interactions, index, model,
                                  user_features, item_features, meta, n_users=5000)

    # Feature engineering
    user_stats = build_user_stats(train_tx)
    item_stats  = build_item_stats(train_tx)
    feat_df, feature_cols = build_ranking_features(pairs_df, user_stats, item_stats,
                                                    item_features, user_features)

    X = feat_df[feature_cols].values
    y = feat_df["label"].values
    # LambdaRank requires group sizes (number of candidates per query)
    group = feat_df.groupby("user_idx").size().values

    log.info(f"Training LightGBM LambdaRank on {len(X):,} samples, {len(feature_cols)} features")

    dtrain = lgb.Dataset(X, label=y, group=group, feature_name=feature_cols)
    ranker = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=LGBM_PARAMS["n_estimators"],
                       valid_sets=[dtrain], callbacks=[lgb.log_evaluation(50)])

    # Save
    ranker.save_model(str(ARTIFACT_DIR / "ranker.lgb"))
    with open(ARTIFACT_DIR / "ranker_meta.json", "w") as f:
        json.dump({"feature_cols": feature_cols}, f)

    log.info("Ranker saved.")

    # Quick eval on val
    evaluate(ranker, val_interactions, train_interactions, index, model,
             user_features, item_features, train_tx, meta, feature_cols)


# ── Evaluate ranker ────────────────────────────────────────────────────────

def evaluate(ranker, val_interactions, train_interactions, index, model,
             user_features, item_features, train_tx, meta, feature_cols,
             n_eval: int = 500):
    import random
    user_stats = build_user_stats(train_tx)
    item_stats  = build_item_stats(train_tx)
    val_users  = list(val_interactions.keys())
    sample     = random.sample(val_users, min(n_eval, len(val_users)))
    user_feat_cols = meta["user_feat_cols"]

    ndcgs, recalls = [], []
    for user_idx in sample:
        seen      = set(train_interactions.get(user_idx, []))
        positives = set(val_interactions.get(user_idx, []))
        if not positives:
            continue
        candidates = retrieve(model, user_idx, user_features, user_feat_cols,
                              index, top_k=RETRIEVAL_TOP_K, exclude_seen=seen)
        if not candidates:
            continue

        import torch
        uf = user_features.set_index("user_idx")
        if user_idx not in uf.index:
            continue
        row = uf.loc[user_idx]
        device_t = next(model.parameters()).device
        uid_t = torch.tensor([user_idx], device=device_t)
        uf_t  = {c: torch.tensor([int(row[c])], device=device_t) for c in user_feat_cols}
        with torch.no_grad():
            u_emb = model.get_user_embedding(uid_t, uf_t).cpu().numpy().astype("float32")

        item_emb_matrix = np.load(ARTIFACT_DIR / "item_embeddings.npy")[candidates]
        scores = (u_emb @ item_emb_matrix.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])

        pairs = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates,
                               "retrieval_score": scores.tolist(), "label": 0})
        feat_df, _ = build_ranking_features(pairs, user_stats, item_stats,
                                             item_features, user_features)
        X = feat_df[feature_cols].values
        pred_scores = ranker.predict(X)
        ranked = [candidates[i] for i in np.argsort(-pred_scores)]

        ndcgs.append(ndcg_at_k(positives, ranked, RANKER_TOP_N))
        recalls.append(recall_at_k(positives, ranked, RANKER_TOP_N))

    log.info(f"Ranker eval (n={len(ndcgs)}) — "
             f"NDCG@{RANKER_TOP_N}: {np.mean(ndcgs):.4f} | "
             f"Recall@{RANKER_TOP_N}: {np.mean(recalls):.4f}")


# ── Inference ──────────────────────────────────────────────────────────────

def infer(user_idx: int) -> list[int]:
    """Return top-N ranked item indices for a given user."""
    with open(ARTIFACT_DIR / "model_meta.json") as f:
        meta = json.load(f)
    with open(ARTIFACT_DIR / "ranker_meta.json") as f:
        ranker_meta = json.load(f)
    with open(PROCESSED_DIR / "train_interactions.pkl", "rb") as f:
        train_interactions = pickle.load(f)

    user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")
    train_tx = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    model  = load_model(meta, user_features, item_features)
    index  = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
    ranker = lgb.Booster(model_file=str(ARTIFACT_DIR / "ranker.lgb"))
    feature_cols = ranker_meta["feature_cols"]

    seen       = set(train_interactions.get(user_idx, []))
    candidates = retrieve(model, user_idx, user_features, meta["user_feat_cols"],
                          index, top_k=RETRIEVAL_TOP_K, exclude_seen=seen)

    import torch
    uf = user_features.set_index("user_idx")
    row = uf.loc[user_idx]
    device_t = next(model.parameters()).device
    uid_t = torch.tensor([user_idx], device=device_t)
    uf_t  = {c: torch.tensor([int(row[c])], device=device_t) for c in meta["user_feat_cols"]}
    with torch.no_grad():
        u_emb = model.get_user_embedding(uid_t, uf_t).cpu().numpy().astype("float32")

    item_emb_matrix = np.load(ARTIFACT_DIR / "item_embeddings.npy")[candidates]
    scores = (u_emb @ item_emb_matrix.T).squeeze()
    if scores.ndim == 0:
        scores = np.array([float(scores)])

    pairs = pd.DataFrame({"user_idx": user_idx, "item_idx": candidates,
                           "retrieval_score": scores.tolist(), "label": 0})
    user_stats = build_user_stats(train_tx)
    item_stats  = build_item_stats(train_tx)
    feat_df, _ = build_ranking_features(pairs, user_stats, item_stats,
                                         item_features, user_features)
    X = feat_df[feature_cols].values
    pred_scores = ranker.predict(X)
    ranked = [candidates[i] for i in np.argsort(-pred_scores)]
    return ranked[:RANKER_TOP_N]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--user_idx", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        result = infer(args.user_idx)
        log.info(f"Top-{len(result)} recommendations for user {args.user_idx}: {result}")
