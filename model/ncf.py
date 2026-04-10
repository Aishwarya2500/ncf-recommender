import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)

        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(x).squeeze()