from sklearn.model_selection import KFold
from typing import Callable, Generator, List, Tuple
import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


def relu(x):
    return max(x, 0)


def grad_relu(x):
    return 0 if x <= 0 else 1


def arctan(x):
    return np.arctan(x)


def grad_arctan(x):
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
    return (np.linalg.norm(y - y_pred, axis=1) ** 2).mean()


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
    _activation_vectorized: Callable[[float], float]
    _activation_grad_vectorized: Callable[[float], float]
    _neurons_sums: np.ndarray
    _output: np.ndarray
    _input: np.ndarray

    def __init__(self, input_size: int, output_size: int,
                 activation: Callable[[float], float],
                 activation_grad: Callable[[float], float]):
        self._input_size = input_size
        self._output_size = output_size
        self._activation_vec = activation
        self._activation_grad = activation_grad
        self._activation_vec = np.vectorize(
            activation, signature="()->()")
        self._activation_grad_vec = np.vectorize(
            activation_grad, signature="()->()")
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
        # dq_dy = (
        #     grad_from_previous_layer *
        #     weights_from_previous_layer).sum(
        #     axis=1)
        dq_dy = np.matmul(
            np.transpose(weights_from_previous_layer[:, :-1]),
            grad_from_previous_layer)
        dq_ds = dq_dy * \
            np.transpose(self._activation_grad_vec(self._neurons_sums))
        dq_dweights = np.matmul(dq_ds, self._input)
        return dq_ds, dq_dweights

    def back_propagation_output_layer(self, grad_loss: Callable[[float], float],
                                      expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Używać tylko dla warstwy wyjściowej. Wyznacza gradienty dla następnej warstwy dq_ds
        i gradient funkcji straty po wagach tej warstwy dq_dweights
        """
        dq_dy = grad_loss(self._output, expected)
        dq_ds = dq_dy * self._activation_grad_vec(self._neurons_sums)
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
        self._output = self._activation_vec(self._neurons_sums)
        return self._output

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size

    def get_weights(self):
        return self._weights


@dataclass
class NeuralNetworkHistoryRecord:
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float

    def __str__(self) -> str:
        return (f'TRAIN(acc: {self.accuracy}, loss: {self.loss}) '
                f'VAL(acc: {self.val_accuracy}, loss: {self.val_loss})')


class NeuralNetwork:
    _layers: List[Layer]

    def __init__(self, *layers: Layer):
        self._layers = list(layers)

    def add_layer(self, layer: Layer):
        self._layers.append(layer)

    @ classmethod
    def _to_batches(cls, X: np.ndarray, y: np.ndarray, batch_size: int
                    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = X.shape[0]
        for i in range(0, n_samples, batch_size):
            j = min(n_samples, i + batch_size)
            yield X[i:j, :], y[i:j, :]

    def _fit_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   learn_rate: float, batch_size: int,
                   loss_grad: GradR2nToR) -> None:
        n_layers = len(self._layers)
        n_hidden_layers = len(self._layers) - 1

        for X_batch, y_batch in self._to_batches(X_train, y_train, batch_size):
            sum_dq_dweights = [
                np.zeros_like(l.get_weights()) for l in self._layers]

            for x, y in zip(X_batch, y_batch):
                x: np.ndarray
                y: np.ndarray

                # # convert to vertical vectors
                y = y.reshape(-1, 1)
                x = x.reshape(1, -1)

                # calculate gradient matrix
                y_pred = self.predict(x)
                dq_dweights = [None] * n_layers
                prev_dq_ds, dq_dweights[-1] = self._layers[-1].back_propagation_output_layer(
                    loss_grad, y)
                prev_layer = self._layers[-1]

                for i in range(n_hidden_layers - 1, -1, -1):
                    prev_dq_ds, dq_dweights[i] = self._layers[i].back_propagation(
                        prev_dq_ds,
                        prev_layer.get_weights()
                    )
                    prev_layer = self._layers[i]

                # add to sum
                for i, grad in enumerate(dq_dweights):
                    sum_dq_dweights[i] += grad

            n_samples = X_batch.shape[0]
            mean_dq_dweights = [grad / n_samples for grad in sum_dq_dweights]
            for layer, grad in zip(self._layers, mean_dq_dweights):
                layer.nudge_weights(grad, learn_rate)

    # def _predict_sample(self, x: np.ndarray) -> List[np.ndarray]:
    #     """
    #     for a given vertical sample vector x
    #     returns list of vertical prediction vectors for every layer
    #     """
    #     result = [x]
    #     for layer in self._layers:
    #         result.append(layer.predict(result[-1]))
    #     return result

    def _validate(self, loss: R2nToR,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> NeuralNetworkHistoryRecord:
        y_train_pred = self.predict(X_train)
        y_val_pred = self.predict(X_val)
        return NeuralNetworkHistoryRecord(
            loss(y_train_pred, y_train),
            accuracy_score(y_train, y_train_pred),
            loss(y_val_pred, y_val),
            accuracy_score(y_val, y_val_pred),
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            learn_rate: float,
            epochs: int,
            loss: R2nToR,
            loss_grad: GradR2nToR,
            batch_size: int = 1,
            validation_data: Tuple[np.ndarray, np.ndarray] = None,
            n_validation_splits: int = None) -> List[NeuralNetworkHistoryRecord]:

        if validation_data is not None and n_validation_splits is not None:
            raise ValueError(
                "validation_data and n_validation_splits cannot be both set")
        if validation_data is None and n_validation_splits is None:
            raise ValueError(
                "validation_data or n_validation_splits must be set")

        common_fit = {
            'learn_rate': learn_rate,
            'batch_size': batch_size,
            'loss_grad': loss_grad,
        }

        history: List[NeuralNetworkHistoryRecord] = []

        if n_validation_splits is not None:
            kfold = KFold(n_validation_splits, shuffle=True)
            for i, (train_index, val_index) in enumerate(
                    kfold.split(range(epochs))):
                X_train, y_train = X[train_index, :], y[train_index, :]
                X_val, y_val = X[val_index, :], y[val_index, :]
                self._fit_epoch(X_train, y_train, **common_fit)
                record = self._validate(loss, X_train, y_train, X_val, y_val)
                history.append(record)
                print(f'EPOCH {i+1}:\t{record}')

        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data
            for i in range(epochs):
                self._fit_epoch(X_train, y_train, **common_fit)
                record = self._validate(loss, X_train, y_train, X_val, y_val)
                history.append(record)
                print(f'EPOCH {i+1}:\t{record}')

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = X
        for layer in self._layers:
            y = layer.predict(y)
        return y


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X: np.ndarray
    y: np.ndarray
    y = y.reshape(-1, 1)

    input_dim = X.shape[1]
    hidden_dim = 2
    output_dim = y.shape[1]

    model = NeuralNetwork()
    model.add_layer(Layer(input_dim, hidden_dim, relu, grad_relu))
    model.add_layer(Layer(hidden_dim, output_dim, linear, grad_linear))

    model.fit(
        X,
        y,
        learn_rate=0.4,
        epochs=40,
        loss=mse,
        loss_grad=mse_grad,
        batch_size=4,
        n_validation_splits=5)
