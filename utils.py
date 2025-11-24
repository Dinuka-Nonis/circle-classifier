import matplotlib.pyplot as plt


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