import numpy as np


def relu(x):
    return x * (x>0)
def relu_derivative(x):
    return (x > 0).astype(float)  
