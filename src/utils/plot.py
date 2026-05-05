import matplotlib.pyplot as plt

def plot(x, title=""):
    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], x[:,1], s=1)
    plt.title(title)
    plt.axis("equal")
    plt.show()