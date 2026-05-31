# H&M Fashion Recommendation System

Two-stage recommendation system built on the H&M Personalized Fashion Recommendations dataset (Kaggle).

## Architecture

```
User features ──► User Tower ──► User Embedding (128-d) ──┐
                                                           ├──► Contrastive Loss (training)
Item features ──► Item Tower ──► Item Embedding (128-d) ──┘

Online:
Query user vector ──► FAISS ANN Search ──► Top-500 candidates ──► LightGBM Ranker ──► Top-N
```

## Project Structure

```
hm-recsys/
├── data/
│   └── preprocess.py       # Load and preprocess H&M CSVs
├── models/
│   ├── user_tower.py       # User embedding MLP
│   ├── item_tower.py       # Item embedding MLP
│   └── two_tower.py        # Joint model + contrastive loss
├── retrieval/
│   ├── train.py            # Train the two-tower model
│   └── faiss_index.py      # Build and query FAISS index
├── ranking/
│   ├── features.py         # Feature engineering for ranker
│   └── ranker.py           # LightGBM ranker: train + inference
├── serving/
│   └── api.py              # FastAPI inference endpoint
├── utils/
│   ├── metrics.py          # NDCG, MAP, Recall@K
│   └── logger.py           # Logging setup
├── scripts/
│   ├── run_pipeline.sh     # End-to-end run script
│   └── download_data.sh    # Kaggle dataset download
├── config.py               # All hyperparameters in one place
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download H&M dataset (requires Kaggle API key)
bash scripts/download_data.sh

# 3. Run full pipeline
bash scripts/run_pipeline.sh
```

## Step-by-step

```bash
# Preprocess
python data/preprocess.py

# Train two-tower model
python retrieval/train.py

# Build FAISS index
python retrieval/faiss_index.py --mode build

# Train ranker
python ranking/ranker.py --mode train

# Start API
uvicorn serving.api:app --reload
```
## Results

| Stage | Metric | Value |
|-------|--------|-------|
| Retrieval | Recall@100 | 0.0130 |
| Ranking | NDCG@12 | 0.0062 |
| Ranking | Recall@12 | 0.0056 |

## Kaggle Notebook
Full training run: [View on Kaggle](https://www.kaggle.com/code/aarushii26/notebook7e9030ac57)
