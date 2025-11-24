import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight initialization (small random values)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Hidden layer
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        eps = 1e-8  # avoid log(0)
        loss = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        return loss

    def backward(self, X, y):
        """Backpropagation"""
        m = X.shape[0]

        # ---- Output layer gradients ----
        dz2 = self.a2 - y                     # (m, 1)
        dW2 = (self.a1.T @ dz2) / m           # (hidden, 1)
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # ---- Hidden layer gradients ----
        da1 = dz2 @ self.W2.T                # (m, hidden)
        dz1 = da1 * (self.a1 * (1 - self.a1))  # sigmoid derivative
        dW1 = (X.T @ dz1) / m                # (2, hidden)
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # ---- Update weights ----
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
