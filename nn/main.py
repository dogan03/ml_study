import numpy as np
import pandas as pd

from activations import *
from losses import mse_loss
from nn import NN

if __name__ == "__main__":
    housing_df = pd.read_csv("data/Housing.csv")
    housing_df.drop(axis=0, columns=["mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"], inplace=True)
    housing_np = housing_df.to_numpy()
    x = housing_np[:,1:]
    y = housing_np[:,0]
    y = y.reshape(-1,1)
    input_shape = x.shape[1]
    output_shape = 1

    nn = NN(input_shape, 1, 256)
    
    EPOCHS = 10000
    for epoch in range(EPOCHS):

        y_hat, z = nn.forward(x)
        loss = mse_loss(y_hat,y)
        nn.backward(x, y, y_hat, z)
        print("Epoch ", epoch, " Loss: ", loss)