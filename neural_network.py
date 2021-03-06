from typing import Callable, Generator, List, Tuple
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from numpy.random import uniform
from functools import partial
from timeit import timeit

RToR = Callable[[float], float]
RnToRn = Callable[[np.ndarray], np.ndarray]
R2nToRn = Callable[[np.ndarray, np.ndarray], np.ndarray]

vectorize_RtoR = partial(np.vectorize, signature='()->()', otypes=[np.float])
vectorize_VtoV = partial(np.vectorize, signature='(m)->(m)', otypes=[np.float])


def relu(x: np.ndarray) -> np.ndarray:
    x[x < 0] = 0
    return x


def grad_relu(x: np.ndarray) -> np.ndarray:
    result = np.ones_like(x)
    result[x < 0] = 0
    return result


def arctan(x: np.ndarray) -> np.ndarray:
    return np.arctan(x)


def grad_arctan(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + x**2)


def linear(x: np.ndarray) -> np.ndarray:
    return x


def grad_linear(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def mse(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(y - y_pred, axis=1) ** 2)


def mse_grad(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y - y_pred)


class Layer:
    _input_size: int
    _output_size: int
    _weights: np.ndarray
    _activation: RnToRn
    _activation_grad: RnToRn
    _neurons_sums: np.ndarray
    _output: np.ndarray
    _input: np.ndarray

    def __init__(self, input_size: int, output_size: int,
                 activation: RnToRn,
                 activation_grad: RnToRn, output_inicialization=False):
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._activation_grad = activation_grad
        if output_inicialization:
            self._weights = np.zeros(shape=(output_size, input_size + 1))
        else:
            self._weights = uniform(low=(-1 / np.sqrt(input_size)), high=(
                1 / np.sqrt(input_size)), size=(output_size, input_size + 1))
        self._neurons_sums = np.zeros(shape=(1, output_size))
        self._output = np.zeros(shape=(1, output_size))
        self._input = np.zeros(shape=(1, input_size + 1))
        self._input[0][input_size] = 1.0

    def back_propagation(self, grad_from_previous_layer: np.ndarray,
                         weights_from_previous_layer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        U??ywa?? dla ka??dej warstwy opr??cz wyj??ciowej. Wyznacza gradienty dla nast??pnej warstwy dq_ds
        i gradient funkcji straty po wagach tej warstwy dq_dweights
        """
        weights_from_previous_layer = np.delete(
            weights_from_previous_layer, -1, 1)
        dq_dy = np.matmul(
            grad_from_previous_layer,
            weights_from_previous_layer)
        dq_ds = dq_dy * self._activation_grad(self._neurons_sums)
        dq_dweights = np.matmul(np.transpose(dq_ds), self._input)
        return dq_ds, dq_dweights

    def back_propagation_output_layer(self, grad_loss: R2nToRn,
                                      expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        U??ywa?? tylko dla warstwy wyj??ciowej. Wyznacza gradienty dla nast??pnej warstwy dq_ds
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

    def get_weights(self):
        return self._weights


@vectorize_VtoV
def determine_class(pred: np.ndarray) -> np.ndarray:
    result = np.zeros_like(pred)
    i = np.argmax(pred)
    result[i] = 1
    return result


@dataclass
class NeuralNetworkHistoryRecord:
    train_true: np.ndarray
    train_pred: np.ndarray
    val_true: np.ndarray
    val_pred: np.ndarray
    loss: R2nToRn
    time: float

    def losses(self) -> Tuple[float, float]:
        return (self.loss(self.train_true, self.train_pred).mean(),
                self.loss(self.val_true, self.val_pred).mean())

    def losses_str(self) -> str:
        train_loss, val_loss = self.losses()
        return f'train_loss: {train_loss}, val_loss: {val_loss}'

    def accuracies(self) -> Tuple[float, float]:
        train_pred = determine_class(self.train_pred)
        val_pred = determine_class(self.val_pred)
        return (accuracy_score(self.train_true, train_pred),
                accuracy_score(self.val_true, val_pred))

    def all_str(self, tabs: int = 0) -> str:
        train_loss, val_loss = self.losses()
        train_acc, val_acc = self.accuracies()
        tabs = tabs * "\t"
        return (f'{tabs}train_loss: {train_loss:.5f}, train_acc: {train_acc:.5f}\n'
                f'{tabs}val_loss  : {val_loss:.5f}, val_acc  : {val_acc:.5f}')


def _to_batches(X: np.ndarray, y: np.ndarray, batch_size: int
                ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        j = min(n_samples, i + batch_size)
        yield X[i:j, :], y[i:j, :]


class NeuralNetwork:
    _layers: List[Layer]
    _is_classifier: bool

    def __init__(self, *layers: Layer, is_classifier: bool = False):
        self._layers = list(layers)
        self._is_classifier = is_classifier

    def add_layer(self, layer: Layer):
        self._layers.append(layer)

    def _fit_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   learn_rate: float, batch_size: int,
                   loss_grad: R2nToRn) -> None:

        n_layers = len(self._layers)
        n_hidden_layers = n_layers - 1

        for X_batch, y_batch in _to_batches(X_train, y_train, batch_size):
            self.predict(X_batch)

            dq_dweights = [
                np.zeros_like(l.get_weights()) for l in self._layers]
            prev_dq_ds, dq_dweights[-1] = self._layers[-1].back_propagation_output_layer(
                loss_grad, y_batch)
            prev_layer = self._layers[-1]

            for i in range(n_hidden_layers - 1, -1, -1):
                prev_dq_ds, dq_dweights[i] = self._layers[i].back_propagation(
                    prev_dq_ds,
                    prev_layer.get_weights()
                )
                prev_layer = self._layers[i]

            n_samples = X_batch.shape[0]
            for layer, grad in zip(self._layers, dq_dweights):
                layer.nudge_weights(grad / n_samples, learn_rate)

    def _validate(self, loss: R2nToRn,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  time: float) -> NeuralNetworkHistoryRecord:
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)
        return NeuralNetworkHistoryRecord(
            train_true=y_train,
            train_pred=y_train_pred,
            val_true=y_val,
            val_pred=y_val_pred,
            loss=loss,
            time=time
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            learn_rate: float,
            epochs: int,
            loss: R2nToRn,
            loss_grad: R2nToRn,
            batch_size: int = 1) -> List[NeuralNetworkHistoryRecord]:

        history: List[NeuralNetworkHistoryRecord] = []

        try:
            for i in range(epochs):
                time = timeit(
                    lambda: self._fit_epoch(
                        X_train, y_train,
                        learn_rate=learn_rate,
                        batch_size=batch_size,
                        loss_grad=loss_grad),
                    number=1)
                record = self._validate(
                    loss, X_train, y_train, X_val, y_val, time)
                history.append(record)
                msg = record.all_str(
                    2) if self._is_classifier else record.losses_str()
                print(f'EPOCH {i+1} [{time:.3f}s]:\n{msg}')
        except KeyboardInterrupt:
            pass

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = X
        for layer in self._layers:
            y = layer.predict(y)
        return y
