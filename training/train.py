# experiments/part1_train.py

import torch

def train(model, dataloader, optimizer, device, steps=20000):
    """
    Part 1训练：v-prediction baseline

    目标：
    - 验证flow matching pipeline是否正确
    """

    model.train()

    loader_iter = iter(dataloader)

    for step in range(steps):

        try:
            x = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            x = next(loader_iter)

        x = x.to(device)

        # =========================
        # 1. sample noise
        # =========================
        eps = torch.randn_like(x)

        # =========================
        # 2. sample time
        # =========================
        t = torch.rand(x.shape[0], 1).to(device)

        # =========================
        # 3. forward process
        # z_t = (1 - t)x + tε
        # =========================
        z = (1 - t) * x + t * eps

        # =========================
        # 4. target velocity
        # =========================
        v_target = eps - x

        # =========================
        # 5. model prediction
        # =========================
        v_pred = model(z, t.squeeze())

        # =========================
        # 6. loss
        # =========================
        loss = ((v_pred - v_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"[Part1] step {step} | loss {loss.item():.4f}")

    return model