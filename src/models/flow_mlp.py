import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device

        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )

        args = t[:, None] * freq[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMLP(nn.Module):
    def __init__(self, x_dim=2, hidden=256):
        super().__init__()

        self.t_embed = TimeEmbedding()

        self.net = nn.Sequential(
            nn.Linear(x_dim + 128, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, x_dim)
        )

    def forward(self, x, t):
        t_emb = self.t_embed(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.net(x)