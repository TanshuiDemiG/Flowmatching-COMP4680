import sys
from pathlib import Path
import os

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from src.dataloader import get_dataloader
from src.models.flow_mlp import FlowMLP
from src.training.train import train
from src.sampling.euler import sample
from src.utils.plot import save_plot



# =========================
# config
# =========================
device = "cuda" 
# if torch.cuda.is_available() else "cpu"
OUT_DIR = "src/output/part2"

datasets = ["swiss_roll", "gaussians", "circles"]
dims = [2, 8, 32]

pred_types = ["x", "v"]
loss_types = ["x", "v"]


# =========================
# helper: run one experiment
# =========================
def run_experiment(dataset_name, dim, pred_type, loss_type):

    print(f"\n=== {dataset_name} | D={dim} | pred={pred_type} | loss={loss_type} ===")

    # -------------------------
    # 1. dataloader
    # -------------------------
    loader = get_dataloader(
        name=dataset_name,
        dim=dim,
        batch_size=1024,
        shuffle=True
    )

    # -------------------------
    # 2. model
    # -------------------------
    model = FlowMLP(x_dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # 3. train
    # -------------------------
    model = train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        steps=15000,
        pred_type=pred_type,
        loss_type=loss_type
    )

    # -------------------------
    # 4. sample
    # -------------------------
    samples = sample(model, n=2000, 
                     dim=dim,#！！！！！！！
                     steps=50, device=device)

    # -------------------------
    # 5. save
    # -------------------------
    save_dir = os.path.join(
        OUT_DIR,
        dataset_name
    )

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"D{dim}_{pred_type}_{loss_type}.png"
    )

    save_plot(samples, save_path, title=save_path)

    print(f"[Saved] {save_path}")


# =========================
# main loop (36 experiments)
# =========================
if __name__ == "__main__":

    for dataset_name in datasets:
        for dim in dims:
            for pred_type in pred_types:
                for loss_type in loss_types:

                    run_experiment(
                        dataset_name,
                        dim,
                        pred_type,
                        loss_type
                    )