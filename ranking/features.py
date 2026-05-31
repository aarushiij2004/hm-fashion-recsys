"""
ranking/features.py
────────────────────
Build the feature matrix used to train and serve the LightGBM ranker.

For each (user, candidate_item) pair we compute:
  - Retrieval score (dot product of two-tower embeddings)
  - User-level stats: avg price paid, purchase count, recency
  - Item-level stats: global popularity, avg price, recency of last purchase
  - Cross features: price affinity, category match count
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger(__name__)


def build_user_stats(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate user-level stats from the training transaction log."""
    stats = tx.groupby("user_idx").agg(
        user_purchase_count=("article_id", "count"),
        user_avg_price=("price", "mean"),
        user_max_price=("price", "max"),
        user_last_purchase=("t_dat", "max"),
    ).reset_index()
    max_date = tx["t_dat"].max()
    stats["user_recency_days"] = (max_date - stats["user_last_purchase"]).dt.days
    stats.drop(columns=["user_last_purchase"], inplace=True)
    return stats


def build_item_stats(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level stats from the training transaction log."""
    stats = tx.groupby("item_idx").agg(
        item_purchase_count=("customer_id", "count"),
        item_avg_price=("price", "mean"),
        item_last_purchase=("t_dat", "max"),
    ).reset_index()
    max_date = tx["t_dat"].max()
    stats["item_recency_days"] = (max_date - stats["item_last_purchase"]).dt.days
    stats.drop(columns=["item_last_purchase"], inplace=True)

    # Log-scale popularity (heavy-tail distribution)
    stats["item_log_popularity"] = np.log1p(stats["item_purchase_count"])
    return stats


def build_ranking_features(candidates_df: pd.DataFrame, user_stats: pd.DataFrame,
                            item_stats: pd.DataFrame, item_features: pd.DataFrame,
                            user_features: pd.DataFrame) -> pd.DataFrame:
    """
    candidates_df columns: user_idx, item_idx, retrieval_score
    Returns feature matrix ready for LightGBM.
    """
    df = candidates_df.copy()

    # Merge user stats
    df = df.merge(user_stats, on="user_idx", how="left")

    # Merge item stats
    df = df.merge(item_stats, on="item_idx", how="left")

    # Merge item categorical features (already label-encoded)
    item_cat_cols = ["product_type_name", "colour_group_name",
                     "department_name", "index_name", "garment_group_name"]
    df = df.merge(item_features[["item_idx"] + item_cat_cols], on="item_idx", how="left")

    # Merge user categorical features
    user_cat_cols = ["club_member_status", "fashion_news_frequency", "age_bin", "has_fn"]
    df = df.merge(user_features[["user_idx"] + user_cat_cols], on="user_idx", how="left")

    # Cross features
    df["price_affinity"] = df["user_avg_price"] / (df["item_avg_price"] + 1e-6)
    df["price_affinity"] = df["price_affinity"].clip(0, 5)

    # Fill missing (cold items/users)
    df.fillna(0, inplace=True)

    feature_cols = [
        "retrieval_score",
        "user_purchase_count", "user_avg_price", "user_max_price", "user_recency_days",
        "item_purchase_count", "item_avg_price", "item_recency_days", "item_log_popularity",
        "price_affinity",
    ] + item_cat_cols + user_cat_cols

    return df, feature_cols
