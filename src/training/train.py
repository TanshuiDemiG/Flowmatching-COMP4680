import torch


def train(model, loader, optimizer, device,
          steps=20000,
          pred_type="v",
          loss_type="v"):

    model.train()
    it = iter(loader)

    for step in range(steps):

        try:
            x = next(it)
        except StopIteration:
            it = iter(loader)
            x = next(it)

        x = x.to(device)

        eps = torch.randn_like(x)
        t = torch.rand(x.shape[0], 1).to(device)

        z = (1 - t) * x + t * eps

        # =========================
        # model forward
        # =========================
        out = model(z, t.squeeze())

        # =========================
        # convert prediction
        # =========================
        if pred_type == "x":
            x_pred = out
            v_pred = (z - x_pred) / (t + 1e-5)
        else:
            v_pred = out
            x_pred = z - t * v_pred

        v_target = eps - x

        # =========================
        # loss selection
        # =========================
        if loss_type == "x":
            loss = ((x_pred - x) ** 2).mean()
        else:
            loss = ((v_pred - v_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            print(f"[train] step {step} loss {loss.item():.4f}")

    return model