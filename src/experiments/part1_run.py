import torch

from dataloader import get_dataloader
from models.flow_mlp import FlowMLP
from training.train import train
from sampling.euler import sample
from utils.plot import plot

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# config (Part 1 only)
# =========================
dataset_name = "swiss_roll"
dim = 2

# =========================
# data
# =========================
loader = get_dataloader(
    name=dataset_name,
    dim=dim,
    batch_size=1024
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
# plot
# =========================
plot(samples, title="Part 1 Result")