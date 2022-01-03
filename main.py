from typing import List
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from evaluate import get_metrics, plot_model
from neural_network import (
    NeuralNetwork,
    Layer,
    NeuralNetworkHistoryRecord,
    arctan,
    determine_class,
    grad_arctan,
    mse,
    mse_grad,
    relu,
    grad_relu,
    linear,
    grad_linear)
from matplotlib import pyplot as plt


def plot_history(history: List[NeuralNetworkHistoryRecord]):
    accuracies = np.array([record.accuracies() for record in history])
    losses = np.array([record.losses() for record in history])

    train_accuracies = accuracies[:, 0]
    val_accuracies = accuracies[:, 1]
    train_losses = losses[:, 0]
    val_losses = losses[:, 1]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))
    fig.legend(['train', 'validation'])
    ax_acc: plt.Axes
    ax_loss: plt.Axes

    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.plot(train_accuracies, label='train')
    ax_acc.plot(val_accuracies, label='validation')
    ax_acc.legend()

    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.plot(train_losses, label='train')
    ax_loss.plot(val_losses, label='validation')
    ax_loss.legend()

    fig.tight_layout()


def iris():
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
    hidden_dim = 16
    output_dim = y.shape[1]

    model = NeuralNetwork(is_classifier=True)
    model.add_layer(Layer(input_dim, hidden_dim, relu, grad_relu))
    model.add_layer(Layer(hidden_dim, output_dim, arctan, grad_arctan,
                          output_inicialization=True))

    history = model.fit(
        X,
        y,
        learn_rate=0.0001,
        epochs=200,
        loss=mse,
        loss_grad=mse_grad,
        batch_size=64,
        n_validation_splits=5)

    plot_history(history)
    plt.show()


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

    plot_history(history)
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
               ("train", "test"), "Neural network")
    plt.show()


if __name__ == "__main__":
    minist()
    pass
