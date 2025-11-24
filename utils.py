import numpy as np
import matplotlib.pyplot as plt

def create_circles_data(n_samples=200, noise=0.08, factor=0.6, random_seed=None):
    
    if random_seed is not None:
        np.random.seed(random_seed)

    n_half = n_samples // 2

    r_outer = 1.0 + noise * np.random.randn(n_half)
    theta_outer = 2* np.pi* np.random.rand(n_half)
    x_outer = np.column_stack((r_outer * np.cos(theta_outer),r_outer*np.sin(theta_outer)))
    y_outer = np.ones((n_half, 1))

    r_inner = factor + noise * np.random.randn(n_half)
    theta_inner = 2 * np.pi * np.random.rand(n_half)
    x_inner = np.column_stack((r_inner * np.cos(theta_inner),r_inner * np.sin(theta_inner)))
    y_inner = np.zeros((n_half, 1))

    x = np.vstack((x_inner, x_outer))
    y = np.vstack((y_inner,y_outer))

    perm = np.random.permutation(n_samples)
    x=x[perm]
    y=y[perm]

    return x, y

def plot_data(x, y, ax = None , title = "Data (blue=0, red =1)"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    
    labels = y.flatten()
    ax.scatter(x[:, 0],x[:, 1], c=labels, cmap="bwr", edgecolors="k", s=40)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return ax