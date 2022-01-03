from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.datasets import load_iris
from neural_network import (
    NeuralNetwork,
    Layer,
    mse,
    mse_grad,
    relu,
    grad_relu,
    linear,
    grad_linear)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X: np.ndarray
    y: np.ndarray
    y = y.reshape(-1, 1)

    # shuffle (original is ordered)
    indexes = np.arange(len(X))
    np.random.shuffle(indexes)
    X = X[indexes]
    y = y[indexes]

    y_enc = OneHotEncoder(sparse=False, dtype=np.int)
    y = y_enc.fit_transform(y)

    input_dim = X.shape[1]
    hidden_dim = 2
    output_dim = y.shape[1]

    model = NeuralNetwork(is_classifier=True)
    model.add_layer(Layer(input_dim, hidden_dim, relu, grad_relu))
    model.add_layer(Layer(hidden_dim, output_dim, linear, grad_linear))

    history = model.fit(
        X,
        y,
        learn_rate=0.4,
        epochs=40,
        loss=mse,
        loss_grad=mse_grad,
        batch_size=1,
        n_validation_splits=5)

    pass
