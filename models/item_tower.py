"""
models/item_tower.py
─────────────────────
MLP that maps item feature indices → 128-d normalized embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ItemTower(nn.Module):
    def __init__(self, num_items: int, feature_dims: dict, embedding_dim: int = 128,
                 hidden_dims: list = None, dropout: float = 0.2):
        """
        Args:
            num_items:      vocabulary size for item ID embedding
            feature_dims:   {feature_name: num_categories}
            embedding_dim:  output embedding dimension
            hidden_dims:    MLP hidden layer sizes
            dropout:        dropout rate
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.item_id_emb = nn.Embedding(num_items, 64, padding_idx=0)

        self.feat_embeddings = nn.ModuleDict()
        feat_emb_dim = 16
        for feat_name, n_cats in feature_dims.items():
            self.feat_embeddings[feat_name] = nn.Embedding(n_cats + 1, feat_emb_dim, padding_idx=0)

        input_dim = 64 + feat_emb_dim * len(feature_dims)

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_ids: torch.Tensor, features: dict) -> torch.Tensor:
        """
        Args:
            item_ids:  (B,) long tensor
            features:  {feat_name: (B,) long tensor}
        Returns:
            (B, embedding_dim) L2-normalized embeddings
        """
        x = self.item_id_emb(item_ids)
        for feat_name, emb_layer in self.feat_embeddings.items():
            x = torch.cat([x, emb_layer(features[feat_name])], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)
