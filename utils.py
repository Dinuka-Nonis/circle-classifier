import matplotlib.pyplot as plt
import numpy as np


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
def plot_decision_boundary(model, X, y, ax=None, title="Decision Boundary"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    # Define grid
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict on grid points
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid)
    Z = (probs > 0.5).astype(int).reshape(xx.shape)

    # Plot contour and points
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
    ax.scatter(X[:,0], X[:,1], c=y.flatten(), cmap="bwr", edgecolors="k", s=40)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return ax