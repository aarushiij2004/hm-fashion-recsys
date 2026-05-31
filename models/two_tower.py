"""
models/two_tower.py
────────────────────
Wraps UserTower + ItemTower and implements in-batch contrastive loss.

Loss: InfoNCE / NT-Xent with in-batch negatives.
  - Each (user, item) in the batch is a positive pair.
  - All other items in the batch are negatives.
  - logits = (user_emb @ item_emb.T) / temperature
  - target = diagonal (each user's positive is index i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.user_tower import UserTower
from models.item_tower import ItemTower


class TwoTowerModel(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower, temperature: float = 0.07):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, user_ids, user_feats, item_ids, item_feats):
        """
        Returns:
            loss (scalar), user_emb (B, D), item_emb (B, D)
        """
        user_emb = self.user_tower(user_ids, user_feats)   # (B, D)
        item_emb = self.item_tower(item_ids, item_feats)   # (B, D)
        loss = self.contrastive_loss(user_emb, item_emb)
        return loss, user_emb, item_emb

    def contrastive_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """
        In-batch InfoNCE loss.
        Similarity matrix: (B, B) — user i vs all items in batch.
        Positive pair is on the diagonal.
        """
        B = user_emb.size(0)
        # cosine similarity matrix (already L2-normalized)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=user_emb.device)

        # Symmetric: both user→item and item→user
        loss_ui = F.cross_entropy(logits, labels)
        loss_iu = F.cross_entropy(logits.T, labels)
        return (loss_ui + loss_iu) / 2

    def get_user_embedding(self, user_ids, user_feats):
        with torch.no_grad():
            return self.user_tower(user_ids, user_feats)

    def get_item_embedding(self, item_ids, item_feats):
        with torch.no_grad():
            return self.item_tower(item_ids, item_feats)
