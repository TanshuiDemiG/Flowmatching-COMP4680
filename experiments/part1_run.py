# experiments/part1_run.py

import torch

from dataloader import get_dataloader
from model import FlowMLP

from part1_train import train
from part1_sample import sample
from utils.plot import plot_points


# =========================
# config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_name = "swiss_roll"
dim = 2

# =========================
# data loader
# =========================
loader = get_dataloader(
    name=dataset_name,
    dim=dim,
    batch_size=1024,
    shuffle=True
)

# =========================
# model
# =========================
model = FlowMLP(x_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# train
# =========================
model = train(model, loader, optimizer, device, steps=20000)

# =========================
# sample
# =========================
samples = sample(model, device=device)

# =========================
# plot results
# =========================
plot_points(samples, title="Part 1 Generated Samples")