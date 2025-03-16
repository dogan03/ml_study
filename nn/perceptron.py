import numpy as np

from activations import *


class Perceptron:
    def __init__(self, input_dim, output_dim, lr = 0.001):
        self.w = np.random.rand(input_dim, output_dim)
        self.bias = np.random.rand(1, output_dim)
        self.lr = lr
    
    def forward(self, x):
        z = x @ self.w + self.bias
        y_hat = relu(z)
        return y_hat, z 
    
    def backward(self, x, y, y_hat, z):
        relu_grad = relu_derivative(z)
        error = 2 * (y_hat - y) * relu_grad
        
        grad_w = (x.T @ error) / len(x)
        grad_b = np.mean(error, axis=0, keepdims=True)
        
        grad_w = np.clip(grad_w, -0.5, 0.5)
        grad_b = np.clip(grad_b, -0.5, 0.5)

        self.w -= self.lr * grad_w
        self.bias -= self.lr * grad_b

