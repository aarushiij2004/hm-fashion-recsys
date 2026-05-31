"""
retrieval/faiss_index.py
─────────────────────────
Build a FAISS IVF-Flat index over all item embeddings.
At query time, encode a user → retrieve top-K items.

Usage:
    python retrieval/faiss_index.py --mode build
    python retrieval/faiss_index.py --mode query --user_idx 42
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import faiss
import pickle

from config import (PROCESSED_DIR, ARTIFACT_DIR, EMBEDDING_DIM,
                    FAISS_NLIST, FAISS_NPROBE, RETRIEVAL_TOP_K, DEVICE)
from models.user_tower import UserTower
from models.item_tower import ItemTower
from models.two_tower import TwoTowerModel
from utils.logger import get_logger

log = get_logger(__name__)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


# ── Load model ─────────────────────────────────────────────────────────────

def load_model(meta: dict, user_features: pd.DataFrame,
               item_features: pd.DataFrame) -> TwoTowerModel:
    from config import USER_HIDDEN_DIMS, ITEM_HIDDEN_DIMS, DROPOUT, TEMPERATURE

    user_feat_cols = meta["user_feat_cols"]
    item_feat_cols = meta["item_feat_cols"]
    num_users = meta["num_users"]
    num_items = meta["num_items"]

    user_feat_dims = {c: int(user_features[c].max()) + 1 for c in user_feat_cols}
    item_feat_dims = {c: int(item_features[c].max()) + 1 for c in item_feat_cols}

    user_tower = UserTower(num_users, user_feat_dims, EMBEDDING_DIM, USER_HIDDEN_DIMS, DROPOUT)
    item_tower = ItemTower(num_items, item_feat_dims, EMBEDDING_DIM, ITEM_HIDDEN_DIMS, DROPOUT)
    model = TwoTowerModel(user_tower, item_tower, temperature=TEMPERATURE).to(device)
    model.load_state_dict(torch.load(ARTIFACT_DIR / "two_tower.pt", map_location=device))
    model.eval()
    log.info("Model loaded.")
    return model


# ── Build all item embeddings ───────────────────────────────────────────────

@torch.no_grad()
def compute_item_embeddings(model: TwoTowerModel, item_features: pd.DataFrame,
                             item_feat_cols: list, num_items: int,
                             batch_size: int = 4096) -> np.ndarray:
    log.info(f"Computing embeddings for {num_items:,} items …")
    uf = item_features.set_index("item_idx").reindex(range(num_items))

    all_embs = []
    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        ids = torch.arange(start, end, device=device)
        feats = {c: torch.tensor(uf[c].fillna(0).values[start:end],
                                  dtype=torch.long, device=device)
                 for c in item_feat_cols}
        emb = model.get_item_embedding(ids, feats).cpu().numpy()
        all_embs.append(emb)

    embeddings = np.vstack(all_embs).astype("float32")
    log.info(f"Item embeddings shape: {embeddings.shape}")
    return embeddings


# ── Build FAISS index ──────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    log.info(f"Building IVF-Flat index: dim={d}, nlist={FAISS_NLIST}")

    quantizer = faiss.IndexFlatIP(d)                        # inner product (cosine on L2-normed)
    index = faiss.IndexIVFFlat(quantizer, d, FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)

    faiss.normalize_L2(embeddings)                          # ensure unit norm
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = FAISS_NPROBE

    log.info(f"Index built: {index.ntotal:,} vectors")
    return index


# ── Retrieve ───────────────────────────────────────────────────────────────

def retrieve(model: TwoTowerModel, user_idx: int, user_features: pd.DataFrame,
             user_feat_cols: list, index: faiss.Index, top_k: int = RETRIEVAL_TOP_K,
             exclude_seen: set = None) -> list[int]:
    """
    Given a user_idx, return top_k item indices (excluding seen items).
    """
    uf = user_features.set_index("user_idx")
    if user_idx not in uf.index:
        log.warning(f"user_idx {user_idx} not in features — returning empty")
        return []

    row = uf.loc[user_idx]
    uid_t  = torch.tensor([user_idx], device=device)
    uf_t   = {c: torch.tensor([int(row[c])], device=device) for c in user_feat_cols}

    with torch.no_grad():
        u_emb = model.get_user_embedding(uid_t, uf_t).cpu().numpy().astype("float32")

    faiss.normalize_L2(u_emb)
    scores, indices = index.search(u_emb, top_k + (len(exclude_seen) if exclude_seen else 0))
    candidates = indices[0].tolist()

    if exclude_seen:
        candidates = [i for i in candidates if i not in exclude_seen]

    return candidates[:top_k]


# ── Main ───────────────────────────────────────────────────────────────────

def main(args):
    with open(ARTIFACT_DIR / "model_meta.json") as f:
        meta = json.load(f)

    user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")

    model = load_model(meta, user_features, item_features)

    if args.mode == "build":
        embeddings = compute_item_embeddings(model, item_features,
                                              meta["item_feat_cols"], meta["num_items"])
        np.save(ARTIFACT_DIR / "item_embeddings.npy", embeddings)

        index = build_index(embeddings)
        faiss.write_index(index, str(ARTIFACT_DIR / "faiss.index"))
        log.info("FAISS index saved.")

    elif args.mode == "query":
        index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
        index.nprobe = FAISS_NPROBE

        with open(PROCESSED_DIR / "train_interactions.pkl", "rb") as f:
            train_interactions = pickle.load(f)

        seen = set(train_interactions.get(args.user_idx, []))
        results = retrieve(model, args.user_idx, user_features,
                           meta["user_feat_cols"], index, top_k=RETRIEVAL_TOP_K, exclude_seen=seen)
        log.info(f"Top-{len(results)} candidates for user {args.user_idx}: {results[:20]} …")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "query"], required=True)
    parser.add_argument("--user_idx", type=int, default=0)
    main(parser.parse_args())
