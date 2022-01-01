from sklearn.model_selection import KFold
from typing import Callable, Generator, List, Tuple
import numpy as np
from numpy.core.fromnumeric import shape


def relu(x):
    return np.arctan(x)


def grad_relu(x):
    return 1 / (1 + x**2)


def linear(x):
    return x


def grad_linear(x):
    return 1


def loss(X, Xd):
    return (X - Xd)**2


def grad_loss(X, Xd):
    return 2 * (X - Xd)


RToR = Callable[[float], float]
R2nToR = Callable[[np.ndarray, np.ndarray], float]
GradR2nToR = Callable[[np.ndarray, np.ndarray], np.ndarray]


def mse(y: np.ndarray, y_pred: np.ndarray):
    return np.linalg.norm(y - y_pred) ^ 2


def mse_grad(y: np.ndarray, y_pred: np.ndarray):
    return 2 * (y - y_pred)


class Layer:
    """
    Niesprawdzone (czyli pewnie nie działające): back_propagation, back_propagation_output_layer
    Sprawdzone tylko dla wektorów, nie macierzy: predict
    """
    _input_size: int
    _output_size: int
    _weights: np.ndarray
    _activation: Callable[[float], float]
    _activation_grad: Callable[[float], float]
    _neurons_sums: np.ndarray
    _output: np.ndarray
    _input: np.ndarray

    def __init__(self, input_size: int, output_size: int,
                 activation: Callable[[float], float],
                 activation_grad: Callable[[float], float]):
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._activation_grad = activation_grad
        self._weights = np.zeros(shape=(output_size, input_size + 1))
        self._neurons_sums = np.zeros(shape=(1, output_size))
        self._output = np.zeros(shape=(1, output_size))
        self._input = np.zeros(shape=(1, input_size + 1))
        self._input[0][input_size] = 1.0

    def back_propagation(self, grad_from_previous_layer: np.ndarray,
                         weights_from_previous_layer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Używać dla każdej warstwy oprócz wyjściowej. Wyznacza gradienty dla następnej warstwy dq_ds
        i gradient funkcji straty po wagach tej warstwy dq_dweights
        """
        dq_dy = (
            grad_from_previous_layer *
            weights_from_previous_layer).sum(
            axis=1)
        dq_ds = dq_dy * self._activation_grad(self._neurons_sums)
        dq_dweights = np.matmul(np.transpose(dq_ds), self._input)
        return dq_ds, dq_dweights

    def back_propagation_output_layer(self, grad_loss: Callable[[float], float],
                                      expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Używać tylko dla warstwy wyjściowej. Wyznacza gradienty dla następnej warstwy dq_ds
        i gradient funkcji straty po wagach tej warstwy dq_dweights
        """
        dq_dy = grad_loss(self._output, expected)
        dq_ds = dq_dy * self._activation_grad(self._neurons_sums)
        dq_dweights = np.matmul(np.transpose(dq_ds), self._input)
        return dq_ds, dq_dweights

    def nudge_weights(self, grad: np.ndarray, learning_rate: float):
        self._weights -= grad * learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._input = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self._neurons_sums = np.transpose(
            np.matmul(
                self._weights,
                np.transpose(
                    self._input)))
        self._output = self._activation(self._neurons_sums)
        return self._output

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size


class NeuralNetwork:
    layers: List[Layer]

    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    @ classmethod
    def _to_batches(cls, X: np.ndarray, y: np.ndarray, batch_size: int
                    ) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            j = min(n_samples, i + batch_size)
            yield X[i:j, :], y[i:j, :]

    def _fit_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   learn_rate: float, batch_size: int,
                   loss_grad: GradR2nToR) -> None:
        for X_batch, y_batch in self._to_batches(X_train, y_train, batch_size):
            sum_grads = [np.zeros(self.layers[0].get_input_size())] + \
                [np.zeros(layer.get_output_size() for layer in self.layers)]
            for x, y in zip(X_batch, y_batch):
                # convert to vertical vectors
                y = y.reshape(-1, 1)
                x = x.reshape(-1, 1)

                # calculate gradient matrix
                Y_preds = self._predict_sample(x)
                grads = [None] * len(sum_grads)
                grads[-1] = loss_grad(y, Y_preds[-1])
                for i, (y, layer) in enumerate(
                        reversed(zip(Y_preds[:-1], self.layers))):
                    grads[-i - 1] = layer.gradient(y, grads[0]), grads[-i]

                # add to sum
                for i, grad in enumerate(grads):
                    sum_grads[i] += grad

            n_samples = X_batch.shape[0]
            mean_grads = [grad / n_samples for grad in sum_grads]
            for layer, grad in zip(self.layers, mean_grads[1:]):
                layer.nudge_weights(grad, learn_rate)

    def _predict_sample(self, x: np.ndarray) -> List[np.ndarray]:
        """
        for a given vertical sample vector x
        returns list of vertical prediction vectors for every layer
        """
        result = [x]
        for layer in self.layers:
            result.append(layer.predict(result[-1]))
        return result

    def fit(self, X: np.ndarray, y: np.ndarray,
            learn_rate: float,
            epochs: int,
            loss: R2nToR,
            loss_grad: GradR2nToR,
            batch_size: int = 1,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            n_validation_splits: int = None) -> None:

        if validation_data is not None and n_validation_splits is not None:
            raise ValueError(
                "validation_data and n_validation_splits cannot be both set")
        if validation_data is None and n_validation_splits is None:
            raise ValueError(
                "validation_data or n_validation_splits must be set")

        common = {
            'learn_rate': learn_rate,
            'batch_size': batch_size,
            'loss_grad': loss_grad,
        }

        if n_validation_splits is not None:
            kfold = KFold(n_validation_splits)
            for train_index, val_index in kfold.split(range(epochs)):
                X_train, y_train = X[train_index, :], y[train_index, :]
                X_val, y_val = X[val_index, :], y[val_index, :]
                self._fit_epoch(X_train, y_train, **common)
                # TODO: validate
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data
            for _ in range(epochs):
                self._fit_epoch(X_train, y_train, **common)
                # TODO: validate

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_sample(X)[-1]
