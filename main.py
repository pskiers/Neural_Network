from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from evaluate import get_metrics, plot_model, plot_history
from neural_network import (
    NeuralNetwork,
    Layer,
    arctan,
    determine_class,
    grad_arctan,
    mse,
    mse_grad,
    relu,
    grad_relu)
from matplotlib import pyplot as plt


def minist():
    X, y = load_digits(return_X_y=True)
    X: np.ndarray
    y: np.ndarray
    y = y.reshape(-1, 1)

    y_enc = OneHotEncoder(sparse=False, dtype=np.int)
    y = y_enc.fit_transform(y)

    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25)

    input_dim = X.shape[1]
    hidden_dims = [1024, 512, 256, 128]
    output_dim = y.shape[1]

    model = NeuralNetwork(is_classifier=True)
    model.add_layer(Layer(input_dim, hidden_dims[0], relu, grad_relu))
    for hidden_dim1, hidden_dim2 in zip(hidden_dims[0:-1], hidden_dims[1:]):
        model.add_layer(Layer(hidden_dim1, hidden_dim2, relu, grad_relu))
    model.add_layer(Layer(hidden_dims[-1], output_dim, arctan, grad_arctan,
                          output_inicialization=True))

    history = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        learn_rate=0.05,
        epochs=10000,
        loss=mse,
        loss_grad=mse_grad,
        batch_size=128)

    total_time = sum(record.time for record in history)
    mean_time = total_time / len(history)

    plt_title = f'Neural network \nTotal time: {total_time:.1f}s; Time per epoch: {mean_time:.3f}s'

    plot_history(history, plt_title)
    plt.show()

    y_train_pred = determine_class(model.predict(X_train))
    y_test_pred = determine_class(model.predict(X_test))

    y_train = y_enc.inverse_transform(y_train)
    y_train_pred = y_enc.inverse_transform(y_train_pred)
    y_test = y_enc.inverse_transform(y_test)
    y_test_pred = y_enc.inverse_transform(y_test_pred)

    train_metrics, train_cm = get_metrics(y_train, y_train_pred)
    test_metrics, test_cm = get_metrics(y_test, y_test_pred)

    plot_model((train_metrics, test_metrics), (train_cm, test_cm),
               ("train", "test"), plt_title)
    plt.show()


if __name__ == "__main__":
    minist()
    pass
