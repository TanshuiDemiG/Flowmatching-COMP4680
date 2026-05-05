import torch

def sample(model, n=2000, steps=50, device="cuda"):
    model.eval()

    z = torch.randn(n, 2).to(device)

    dt = 1.0 / steps

    with torch.no_grad():
        for i in range(steps):

            t = torch.ones(n, 1).to(device) * (1 - i / steps)

            v = model(z, t.squeeze())

            z = z - v * dt

    return z.cpu()