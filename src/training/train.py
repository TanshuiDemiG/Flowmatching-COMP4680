import torch

def train(model, loader, optimizer, device, steps=20000):
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

        v_target = eps - x
        v_pred = model(z, t.squeeze())

        loss = ((v_pred - v_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"[train] step {step} loss {loss.item():.4f}")

    return model