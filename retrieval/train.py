"""
retrieval/train.py
───────────────────
Train the two-tower model on H&M interaction data.

Usage:
    python retrieval/train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (PROCESSED_DIR, ARTIFACT_DIR, EMBEDDING_DIM, BATCH_SIZE,
                    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE, TEMPERATURE,
                    NUM_WORKERS, DEVICE, USER_HIDDEN_DIMS, ITEM_HIDDEN_DIMS, DROPOUT)
from models.user_tower import UserTower
from models.item_tower import ItemTower
from models.two_tower import TwoTowerModel
from utils.metrics import recall_at_k
from utils.logger import get_logger

log = get_logger(__name__)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")


# ── Dataset ────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    Each sample: one (user, positive_item) pair.
    User and item features are looked up from feature tables.
    """
    def __init__(self, interactions: dict, user_features: pd.DataFrame,
                 item_features: pd.DataFrame, user_feat_cols: list, item_feat_cols: list):
        self.pairs = [(u, i) for u, items in interactions.items() for i in items]
        self.user_feat_cols = user_feat_cols
        self.item_feat_cols = item_feat_cols

        # Index feature tables by idx for O(1) lookup
        self.user_feat = user_features.set_index("user_idx")
        self.item_feat = item_features.set_index("item_idx")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_idx, item_idx = self.pairs[idx]

        u_row = self.user_feat.loc[user_idx]
        i_row = self.item_feat.loc[item_idx]

        user_feats = {c: int(u_row[c]) for c in self.user_feat_cols}
        item_feats = {c: int(i_row[c]) for c in self.item_feat_cols}

        return {
            "user_id": user_idx,
            "item_id": item_idx,
            "user_feats": user_feats,
            "item_feats": item_feats,
        }


def collate_fn(batch):
    user_ids = torch.tensor([b["user_id"] for b in batch], dtype=torch.long)
    item_ids = torch.tensor([b["item_id"] for b in batch], dtype=torch.long)
    user_feat_keys = list(batch[0]["user_feats"].keys())
    item_feat_keys = list(batch[0]["item_feats"].keys())
    user_feats = {k: torch.tensor([b["user_feats"][k] for b in batch], dtype=torch.long)
                  for k in user_feat_keys}
    item_feats = {k: torch.tensor([b["item_feats"][k] for b in batch], dtype=torch.long)
                  for k in item_feat_keys}
    return user_ids, item_ids, user_feats, item_feats


# ── Build model ────────────────────────────────────────────────────────────

def build_model(user_features: pd.DataFrame, item_features: pd.DataFrame,
                user_feat_cols: list, item_feat_cols: list,
                num_users: int, num_items: int) -> TwoTowerModel:

    user_feat_dims = {c: int(user_features[c].max()) + 1 for c in user_feat_cols}
    item_feat_dims = {c: int(item_features[c].max()) + 1 for c in item_feat_cols}

    user_tower = UserTower(num_users, user_feat_dims, EMBEDDING_DIM, USER_HIDDEN_DIMS, DROPOUT)
    item_tower = ItemTower(num_items, item_feat_dims, EMBEDDING_DIM, ITEM_HIDDEN_DIMS, DROPOUT)
    model = TwoTowerModel(user_tower, item_tower, temperature=TEMPERATURE)
    return model.to(device)


# ── Evaluation (Recall@K on val set using brute-force dot product) ─────────

@torch.no_grad()
def evaluate(model, val_interactions: dict, user_features: pd.DataFrame,
             item_features: pd.DataFrame, user_feat_cols: list,
             item_feat_cols: list, num_items: int, k: int = 100) -> float:
    model.eval()

    # Build all item embeddings
    all_item_ids = torch.arange(num_items, device=device)
    item_feat_tensors = {
        c: torch.tensor(item_features.set_index("item_idx").reindex(
            range(num_items))[c].fillna(0).values, dtype=torch.long, device=device)
        for c in item_feat_cols
    }
    all_item_emb = model.get_item_embedding(all_item_ids, item_feat_tensors)  # (N_items, D)

    uf = user_features.set_index("user_idx")
    recalls = []

    # Sample 1000 users for speed
    val_users = list(val_interactions.keys())
    sample_users = random.sample(val_users, min(1000, len(val_users)))

    for user_idx in sample_users:
        if user_idx not in uf.index:
            continue
        row = uf.loc[user_idx]
        uid_t  = torch.tensor([user_idx], device=device)
        uf_t   = {c: torch.tensor([int(row[c])], device=device) for c in user_feat_cols}
        u_emb  = model.get_user_embedding(uid_t, uf_t)       # (1, D)
        scores = (u_emb @ all_item_emb.T).squeeze(0)         # (N_items,)
        top_k  = scores.topk(k).indices.cpu().numpy().tolist()
        positives = set(val_interactions[user_idx])
        recalls.append(recall_at_k(positives, top_k, k))

    return float(np.mean(recalls))


# ── Training loop ──────────────────────────────────────────────────────────

def train():
    log.info("Loading processed data …")
    user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")

    with open(PROCESSED_DIR / "train_interactions.pkl", "rb") as f:
        train_interactions = pickle.load(f)
    with open(PROCESSED_DIR / "val_interactions.pkl", "rb") as f:
        val_interactions = pickle.load(f)
    with open(PROCESSED_DIR / "user_enc.pkl", "rb") as f:
        user_enc = pickle.load(f)
    with open(PROCESSED_DIR / "item_enc.pkl", "rb") as f:
        item_enc = pickle.load(f)

    num_users = len(user_enc.classes_)
    num_items = len(item_enc.classes_)

    USER_FEAT_COLS = ["club_member_status", "fashion_news_frequency", "age_bin", "has_fn"]
    ITEM_FEAT_COLS = ["product_type_name", "colour_group_name",
                      "department_name", "index_name", "garment_group_name"]

    dataset = InteractionDataset(train_interactions, user_features, item_features,
                                 USER_FEAT_COLS, ITEM_FEAT_COLS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = build_model(user_features, item_features, USER_FEAT_COLS, ITEM_FEAT_COLS,
                        num_users, num_items)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    log.info(f"Training: {len(dataset):,} pairs | {num_users:,} users | {num_items:,} items")

    best_recall = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, (user_ids, item_ids, user_feats, item_feats) in enumerate(loader):
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_feats = {k: v.to(device) for k, v in user_feats.items()}
            item_feats = {k: v.to(device) for k, v in item_feats.items()}

            optimizer.zero_grad()
            loss, _, _ = model(user_ids, user_feats, item_ids, item_feats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 200 == 0:
                log.info(f"  Epoch {epoch} | step {step+1}/{len(loader)} | loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(loader)

        recall = evaluate(model, val_interactions, user_features, item_features,
                          USER_FEAT_COLS, ITEM_FEAT_COLS, num_items, k=100)
        log.info(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | Recall@100 {recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            patience_counter = 0
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ARTIFACT_DIR / "two_tower.pt")
            log.info(f"  ✓ Saved best model (Recall@100={best_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"Training complete. Best Recall@100: {best_recall:.4f}")

    # Save feature column names for downstream use
    import json
    meta = {"user_feat_cols": USER_FEAT_COLS, "item_feat_cols": ITEM_FEAT_COLS,
            "num_users": num_users, "num_items": num_items}
    with open(ARTIFACT_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    train()
