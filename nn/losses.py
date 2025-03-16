import numpy as np


def mse_loss(y_hat, y):
    summed = (y - y_hat) ** 2
    summed = np.sum(summed)
    avg = summed/len(y)
    return avg