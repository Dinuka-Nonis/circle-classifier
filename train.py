import numpy as np
import matplotlib.pyplot as plt
from data import create_circles_data
from utils import plot_decision_boundary
from nn_numpy import NeuralNetwork

def main():
    # --- 1. Create dataset ---
    X, y = create_circles_data(n_samples=500, noise=0.08, factor=0.6, random_seed=42)

    # --- 2. Create network ---
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5, seed=1)

    epochs = 10000
    loss_list = []

    # --- 3. Training loop ---
    for epoch in range(epochs):
        # forward pass
        y_pred = nn.forward(X)

        # compute loss
        loss = nn.compute_loss(y, y_pred)
        loss_list.append(loss)

        # backward pass
        nn.backward(X, y)

        # print loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # --- 4. Plot loss curve ---
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # --- 5. Plot decision boundary ---
    plt.figure()
    plot_decision_boundary(nn, X, y)
    plt.show()

if __name__ == "__main__":
    main()
