import matplotlib.pyplot as plt
import os


def save_plot(x, path, title=""):
    """
    统一保存所有Part1图像

    参数：
    x: [N,2]
    path: 保存路径
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], x[:,1], s=1)

    plt.title(title)
    plt.axis("equal")

    plt.savefig(path, dpi=200)
    plt.close()