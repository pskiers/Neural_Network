from typing import Callable, List, Tuple
import numpy as np
from numpy.core.fromnumeric import shape


def relu(x):
    return np.arctan(x)


def grad_relu(x):
    return 1/(1+x**2)


def linear(x):
    return x


def grad_linear(x):
    return 1


def loss(X, Xd):
    return (X - Xd)**2


def grad_loss(X, Xd):
    return 2 * (X - Xd)


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
        self._weights = np.zeros(shape=(output_size, input_size+1))
        self._neurons_sums = np.zeros(shape=(1, output_size))
        self._output = np.zeros(shape=(1, output_size))
        self._input = np.zeros(shape=(1, input_size+1))
        self._input[0][input_size] = 1.0

    def back_propagation(self, grad_from_previous_layer: np.ndarray,
                         weights_from_previous_layer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Używać dla każdej warstwy oprócz wyjściowej. Wyznacza gradienty dla następnej warstwy dq_ds
        i gradient funkcji straty po wagach tej warstwy dq_dweights
        """
        dq_dy = (grad_from_previous_layer * weights_from_previous_layer).sum(axis=1)
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
        self._input = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        self._neurons_sums = np.transpose(np.matmul(self._weights, np.transpose(self._input)))
        self._output = self._activation(self._neurons_sums)
        return self._output



class NeuralNetwork:
    layers: List[Layer]

    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def _fit_epoch(self, X: np.ndarray, y: np.ndarray,
                   learn_rate: float, batch_size: int,
                   loss: Callable[[float], float],
                   loss_grad: Callable[[float], float]) -> None:
        pass

    def _predict_epoch(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray,
            learn_rate: float,
            batch_size: int = 1,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            validation_split: float = None) -> None:

        if validation_data is not None and validation_split is not None:
            raise ValueError(
                "validation_data and validation_split cannot be both set")
        if validation_data is None and validation_split is None:
            raise ValueError(
                "validation_data or validation_split must be set")

        for _ in range(batch_size):
            if validation_split is not None:
                pass
            self._fit_epoch(X, y, learn_rate, batch_size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict_epoch(X)[:-1]
