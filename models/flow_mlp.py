import torch
import torch.nn as nn
import math

# =========================
# Time Embedding (sinusoidal)
# =========================
class TimeEmbedding(nn.Module):
    """
    为什么要这个？
    ----------------
    diffusion模型必须知道当前时间 t
    否则同一个 z_t 无法区分“噪声程度”
    """
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device

        # frequency decay
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )

        # shape: [batch, dim/2]
        args = t[:, None] * freq[None]

        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# =========================
# Flow MLP (core model)
# =========================
class FlowMLP(nn.Module):
    """
    输入: (z_t, t)
    输出: v (velocity prediction)
    """

    def __init__(self, x_dim=2, hidden=256):
        super().__init__()

        self.time_emb = TimeEmbedding()

        self.net = nn.Sequential(
            nn.Linear(x_dim + 128, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, x_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.net(x)