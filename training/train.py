# import torch
# from model import FlowMLP
# from data import ToyDataset

# # =========================
# # Training config (Part 1 only)
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = ToyDataset("swiss")
# model = FlowMLP().to(device)

# opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# # =========================
# # Training loop
# # =========================
# for step in range(25000):

#     # -------------------------
#     # 1. sample data
#     # -------------------------
#     x = dataset.sample(1024).to(device)

#     # -------------------------
#     # 2. sample noise
#     # -------------------------
#     eps = torch.randn_like(x)

#     # -------------------------
#     # 3. sample time t
#     # -------------------------
#     t = torch.rand(x.shape[0], 1).to(device)

#     # -------------------------
#     # 4. forward process
#     # z_t = (1-t)x + tε
#     # -------------------------
#     z = (1 - t) * x + t * eps

#     # -------------------------
#     # 5. target (v-pred)
#     # v = ε - x
#     # -------------------------
#     v_target = eps - x

#     # -------------------------
#     # 6. model prediction
#     # -------------------------
#     v_pred = model(z, t.squeeze())

#     # -------------------------
#     # 7. loss
#     # -------------------------
#     loss = ((v_pred - v_target) ** 2).mean()

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     if step % 1000 == 0:
#         print(f"step {step}, loss {loss.item():.4f}")
from dataloader import get_dataloader
from model import FlowMLP

import torch

device = "cuda"

loader = get_dataloader(
    name="swiss_roll",
    dim=2,
    batch_size=1024
)

model = FlowMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(20000):

    for x in loader:
        x = x.to(device)

        eps = torch.randn_like(x)
        t = torch.rand(x.shape[0], 1).to(device)

        z = (1 - t) * x + t * eps

        v_pred = model(z, t.squeeze())
        v_target = eps - x

        loss = ((v_pred - v_target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        break  # 关键：dataloader循环只取一个batch