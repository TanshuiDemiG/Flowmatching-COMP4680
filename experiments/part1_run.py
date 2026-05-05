from dataloader import get_dataloader
from training.train import train
from sampling.euler import sample
from utils.plot import plot

config = {
    "dataset": "swiss_roll",
    "dim": 2,
    "steps": 20000
}

# 1. load data
loader = get_dataloader("swiss_roll", dim=2)

# 2. train
model = train(loader, config)

# 3. sample
samples = sample(model)

# 4. plot
plot(samples)