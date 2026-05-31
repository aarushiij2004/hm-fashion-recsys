"""
models/user_tower.py
─────────────────────
MLP that maps user feature indices → 128-d normalized embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, feature_dims: dict, embedding_dim: int = 128,
                 hidden_dims: list = None, dropout: float = 0.2):
        """
        Args:
            num_users:      vocabulary size for user ID embedding
            feature_dims:   {feature_name: num_categories} for each categorical feature
            embedding_dim:  output embedding dimension
            hidden_dims:    MLP hidden layer sizes
            dropout:        dropout rate
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # User ID embedding (learnable lookup)
        self.user_id_emb = nn.Embedding(num_users, 64, padding_idx=0)

        # Categorical feature embeddings (each gets a small lookup table)
        self.feat_embeddings = nn.ModuleDict()
        feat_emb_dim = 16
        for feat_name, n_cats in feature_dims.items():
            self.feat_embeddings[feat_name] = nn.Embedding(n_cats + 1, feat_emb_dim, padding_idx=0)

        # Total input dim: user_id_emb + all feature embeddings
        input_dim = 64 + feat_emb_dim * len(feature_dims)

        # MLP
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, features: dict) -> torch.Tensor:
        """
        Args:
            user_ids:  (B,) long tensor of user indices
            features:  {feat_name: (B,) long tensor}
        Returns:
            (B, embedding_dim) L2-normalized embeddings
        """
        x = self.user_id_emb(user_ids)                         # (B, 64)
        for feat_name, emb_layer in self.feat_embeddings.items():
            x = torch.cat([x, emb_layer(features[feat_name])], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)                     # unit sphere
