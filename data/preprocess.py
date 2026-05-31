"""
data/preprocess.py
──────────────────
Load raw H&M CSVs, clean, encode categoricals, build interaction matrix,
and split into train / val / test by time.

H&M files expected in DATA_DIR:
  transactions_train.csv  – t_dat, customer_id, article_id, price, sales_channel_id
  customers.csv           – customer_id, age, club_member_status, fashion_news_frequency
  articles.csv            – article_id, product_type_name, colour_group_name,
                            department_name, index_name, garment_group_name
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

from config import DATA_DIR, PROCESSED_DIR, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS, VAL_WEEKS, TEST_WEEKS
from utils.logger import get_logger

log = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def encode_column(df: pd.DataFrame, col: str, encoder: LabelEncoder | None = None):
    df[col] = df[col].astype(str).fillna("unknown")
    if encoder is None:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    else:
        # handle unseen labels
        known = set(encoder.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "unknown")
        df[col] = encoder.transform(df[col])
    return df, encoder


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info(f"Saved → {path}")


# ── Load ───────────────────────────────────────────────────────────────────

def load_raw():
    log.info("Loading raw CSVs …")
    tx = pd.read_csv(DATA_DIR / "transactions_train.csv", parse_dates=["t_dat"])
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    articles = pd.read_csv(DATA_DIR / "articles.csv")
    log.info(f"  transactions: {len(tx):,}")
    log.info(f"  customers:    {len(customers):,}")
    log.info(f"  articles:     {len(articles):,}")
    return tx, customers, articles


# ── Filter ─────────────────────────────────────────────────────────────────

def filter_interactions(tx: pd.DataFrame) -> pd.DataFrame:
    log.info("Filtering sparse users / items …")
    for _ in range(3):   # iterative until stable
        n_before = len(tx)
        user_counts = tx["customer_id"].value_counts()
        item_counts = tx["article_id"].value_counts()
        tx = tx[
            tx["customer_id"].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index) &
            tx["article_id"].isin(item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index)
        ]
        if len(tx) == n_before:
            break
    log.info(f"  {len(tx):,} interactions | {tx['customer_id'].nunique():,} users | {tx['article_id'].nunique():,} items")
    return tx


# ── Feature engineering ────────────────────────────────────────────────────

def build_user_features(customers: pd.DataFrame, user_ids_keep: set) -> pd.DataFrame:
    df = customers[customers["customer_id"].isin(user_ids_keep)].copy()

    # Age: bin into groups
    df["age"] = df["age"].fillna(df["age"].median())
    df["age_bin"] = pd.cut(df["age"], bins=[0,20,30,40,50,60,120],
                           labels=["<20","20s","30s","40s","50s","60+"])

    cat_cols = ["club_member_status", "fashion_news_frequency", "age_bin"]
    encoders = {}
    for c in cat_cols:
        df, enc = encode_column(df, c)
        encoders[c] = enc

    df["has_fn"] = (df["fashion_news_frequency"] > 0).astype(int)
    keep = ["customer_id"] + cat_cols + ["has_fn"]
    return df[keep].reset_index(drop=True), encoders


def build_item_features(articles: pd.DataFrame, item_ids_keep: set) -> pd.DataFrame:
    df = articles[articles["article_id"].isin(item_ids_keep)].copy()

    cat_cols = ["product_type_name", "colour_group_name",
                "department_name", "index_name", "garment_group_name"]
    encoders = {}
    for c in cat_cols:
        df, enc = encode_column(df, c)
        encoders[c] = enc

    keep = ["article_id"] + cat_cols
    return df[keep].reset_index(drop=True), encoders


# ── Train / val / test split (time-based) ──────────────────────────────────

def time_split(tx: pd.DataFrame):
    max_date = tx["t_dat"].max()
    test_start  = max_date - pd.Timedelta(weeks=TEST_WEEKS)
    val_start   = test_start - pd.Timedelta(weeks=VAL_WEEKS)

    train = tx[tx["t_dat"] < val_start]
    val   = tx[(tx["t_dat"] >= val_start) & (tx["t_dat"] < test_start)]
    test  = tx[tx["t_dat"] >= test_start]

    # only keep val/test users/items seen in train
    train_users = set(train["customer_id"])
    train_items = set(train["article_id"])
    val  = val[val["customer_id"].isin(train_users) & val["article_id"].isin(train_items)]
    test = test[test["customer_id"].isin(train_users) & test["article_id"].isin(train_items)]

    log.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


# ── ID encoders ────────────────────────────────────────────────────────────

def encode_ids(tx_train, tx_val, tx_test):
    user_enc = LabelEncoder().fit(list(tx_train["customer_id"].unique()))
    item_enc = LabelEncoder().fit(list(tx_train["article_id"].unique()))

    for df in [tx_train, tx_val, tx_test]:
        df["user_idx"] = user_enc.transform(df["customer_id"])
        df["item_idx"] = item_enc.transform(df["article_id"])

    return tx_train, tx_val, tx_test, user_enc, item_enc


# ── Build positive pairs for contrastive training ──────────────────────────

def build_interaction_dict(tx: pd.DataFrame) -> dict:
    """user_idx → sorted list of item_idx they interacted with"""
    return tx.groupby("user_idx")["item_idx"].apply(sorted).to_dict()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    tx, customers, articles = load_raw()
    tx = filter_interactions(tx)

    train, val, test = time_split(tx)
    train, val, test, user_enc, item_enc = encode_ids(train, val, test)

    user_features, user_feat_enc = build_user_features(customers, set(train["customer_id"]))
    item_features, item_feat_enc = build_item_features(articles, set(train["article_id"]))

    # Merge encoded IDs into feature tables
    user_id_map = pd.DataFrame({"customer_id": user_enc.classes_,
                                 "user_idx": range(len(user_enc.classes_))})
    item_id_map = pd.DataFrame({"article_id": item_enc.classes_,
                                 "item_idx": range(len(item_enc.classes_))})

    user_features = user_features.merge(user_id_map, on="customer_id", how="inner")
    item_features = item_features.merge(item_id_map, on="article_id", how="inner")

    train_interactions = build_interaction_dict(train)
    val_interactions   = build_interaction_dict(val)
    test_interactions  = build_interaction_dict(test)

    # ── Save ──
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    user_features.to_parquet(PROCESSED_DIR / "user_features.parquet", index=False)
    item_features.to_parquet(PROCESSED_DIR / "item_features.parquet", index=False)

    save(user_enc,          PROCESSED_DIR / "user_enc.pkl")
    save(item_enc,          PROCESSED_DIR / "item_enc.pkl")
    save(user_feat_enc,     PROCESSED_DIR / "user_feat_encoders.pkl")
    save(item_feat_enc,     PROCESSED_DIR / "item_feat_encoders.pkl")
    save(train_interactions, PROCESSED_DIR / "train_interactions.pkl")
    save(val_interactions,   PROCESSED_DIR / "val_interactions.pkl")
    save(test_interactions,  PROCESSED_DIR / "test_interactions.pkl")

    log.info("Preprocessing complete.")
    log.info(f"  Users: {len(user_enc.classes_):,}  Items: {len(item_enc.classes_):,}")


if __name__ == "__main__":
    main()
