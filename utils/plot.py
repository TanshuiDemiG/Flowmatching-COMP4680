# import os
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_samples(gt_samples, generated_samples, save_path, title=""):
#     """
#     Plots ground truth samples vs generated samples in 2D.
#     gt_samples: (N, 2)
#     generated_samples: (N, 2)
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.figure(figsize=(10, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.scatter(gt_samples[:, 0], gt_samples[:, 1], alpha=0.5, s=2)
#     plt.title("Ground Truth (2D)")
#     plt.axis('equal')
    
#     plt.subplot(1, 2, 2)
#     plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, s=2, color='orange')
#     plt.title("Generated (2D)")
#     plt.axis('equal')
    
#     if title:
#         plt.suptitle(title)
        
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def plot_loss_curve(losses, save_path, title="Training Loss"):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.figure(figsize=(6, 4))
#     plt.plot(losses)
#     plt.title(title)
#     plt.xlabel("Steps")
#     plt.ylabel("Loss")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()


import matplotlib.pyplot as plt

# =========================
# visualization
# =========================

def plot(x, title=""):
    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0].cpu(), x[:,1].cpu(), s=1)
    plt.title(title)
    plt.show()