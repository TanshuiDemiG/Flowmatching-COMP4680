import sys
from pathlib import Path
import os

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import get_dataloader
from src.models.flow_mlp import FlowMLP
from src.training.train import train
from src.sampling.euler import sample
from src.utils.plot import save_plot

device = "cuda" if torch.cuda.is_available() else "cpu"


OUT_DIR = PROJECT_ROOT / "src" / "output" / "part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)



# =========================
# config 
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"


OUT_DIR = PROJECT_ROOT / "src" / "output" / "part1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = ["swiss_roll", "gaussians", "circles"]
for dataset_name in datasets:
    dim = 2

# dataset_name = "swiss_roll"
# dim = 2








# =========================
# run
# =========================



for name in datasets:

    print(f"\n===== Running Part 1 on {name} =====")

    # =========================
    # 1. dataloader
    # =========================
    loader = get_dataloader(
        name=name,
        dim=2,
        batch_size=1024
    )

    # =========================
    # 2. model
    # =========================
    model = FlowMLP(x_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # =========================
    # 3. train
    # =========================
    model = train(model, loader, optimizer, device, steps=20000)

    # =========================
    # 4. sample
    # =========================
    samples = sample(model, device=device)

    # =========================
    # 5. save plots
    # =========================
    save_plot(
        samples,
        os.path.join(OUT_DIR, name, "generated.png"),
        title=f"{name} generated"
    )

    print(f"[DONE] {name}")