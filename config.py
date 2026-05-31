from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

for d in [DATA_DIR, PROCESSED_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
MIN_USER_INTERACTIONS = 5      # drop users with fewer transactions
MIN_ITEM_INTERACTIONS = 3      # drop items with fewer transactions
VAL_WEEKS = 1                  # last N weeks held out for validation
TEST_WEEKS = 1                 # last N weeks (after val) held out for test

# ── Embedding model ────────────────────────────────────────────────────────
EMBEDDING_DIM = 128
USER_HIDDEN_DIMS = [256, 128]
ITEM_HIDDEN_DIMS = [256, 128]
DROPOUT = 0.2
TEMPERATURE = 0.07             # contrastive loss temperature

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 20
PATIENCE = 3                   # early stopping patience
NUM_WORKERS = 4
DEVICE = "cuda"                # falls back to cpu in train.py

# ── Retrieval ──────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 500          # candidates passed to ranker
FAISS_NLIST = 100              # IVF-Flat: number of Voronoi cells
FAISS_NPROBE = 10              # cells to probe at query time

# ── Ranking ────────────────────────────────────────────────────────────────
RANKER_TOP_N = 12              # final recommendations per user
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_K_VALUES = [10, 20, 50, 100]
