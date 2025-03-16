import numpy as np

from perceptron import Perceptron


class NN:
    def __init__(self, input_dim, output_dim, hidden_layer_dim):
        self.perceptrons = [Perceptron(input_dim, output_dim, lr=0.1) for _ in range(hidden_layer_dim)]


    def forward(self, x):
        y_hat, z = [], []

        for perceptron in self.perceptrons:
            y_h, _z = perceptron.forward(x)
            y_hat.append(y_h)
            z.append(_z)
        y_hat = np.mean(np.array(y_hat), 0)
        z = np.mean(np.array(z), 0)

        return y_hat, z

    def backward(self, x, y, y_hat, z):
        for perceptron in self.perceptrons:
            perceptron.backward(x, y, y_hat, z)
