from typing import Callable, List, Tuple
import numpy as np


class Layer:
    _input_size: int
    _output_size: int
    _weights: np.ndarray

    def __init__(self, input_size: int, output_size: int,
                 activation: Callable[[float], float],
                 activation_grad: Callable[[float], float]):
        pass

    def gradient(self, grad: np.ndarray) -> np.ndarray:
        pass

    def nudge_weights(self, grad: np.ndarray, learn_rate: float):
        self._weights -= grad * learn_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


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
