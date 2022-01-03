from copy import copy
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from neural_network import NeuralNetworkHistoryRecord

PLOT_DIR = "plots"


def prepare_bars(n_bars: int, x: Iterable[float],
                 xtickslbls: Iterable[Union[str, int, float]] = None,
                 ax: Optional[plt.Axes] = None):
    width = 1 / (n_bars + 1)
    offset = (width * i for i in range(n_bars))
    if xtickslbls is None:
        xtickslbls = copy(x)
    tick_offset = - width / 2 + (n_bars / 2 * width)
    tick_positions = {l: xx for xx, l in zip(x + tick_offset, xtickslbls)}
    if ax is not None:
        ax.set_xticks(list(tick_positions.values()))
        ax.set_xticklabels(xtickslbls)
    else:
        plt.xticks(list(tick_positions.values()), xtickslbls)
    return (width, offset, tick_positions)


def get_metrics(y: pd.Series, y_pred: pd.Series
                ) -> Tuple[Dict[str, float], np.ndarray]:
    """calculates metrics and confusion matrix"""
    common = {'average': 'macro',
              'zero_division': 0}
    return (
        {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, **common),
            'recall': recall_score(y, y_pred, **common),
            'f1': f1_score(y, y_pred, **common)
        },
        confusion_matrix(y, y_pred)
    )


def plot_model(
        set_metrics: List[Dict[str, float]],
        set_conf_mat: List[np.ndarray],
        set_names: List[str],
        title: str) -> List[float]:
    n_sets = len(set_metrics)

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(ncols=n_sets + 1, nrows=n_sets, figure=fig)

    ax_bar: plt.Axes = fig.add_subplot(gs[:, 1:])

    x_bar = np.arange(len(set_metrics[0]))
    width, offset, _ = prepare_bars(
        n_sets, x_bar, set_metrics[0].keys(), ax=ax_bar)

    for i, (metrics, conf_mat, set_name) in enumerate(
            zip(set_metrics, set_conf_mat, set_names)):
        ax: plt.Axes = fig.add_subplot(gs[i, 0])
        ax.set_aspect('equal')
        ax.set_title(set_name)

        heatmap(conf_mat / conf_mat.sum() * 100, ax=ax)
        for t in ax.texts:
            t.set_text(t.get_text() + " %")

        x_cur = x_bar + next(offset)
        y_bar = list(metrics.values())
        ax_bar.bar(
            x_cur,
            y_bar,
            label=set_name,
            width=width,
            edgecolor='black')
        for xx, yy in zip(x_cur, y_bar):
            ax_bar.text(xx, yy - 0.02, f'{yy:.3f}', color='white',
                        horizontalalignment='center',
                        verticalalignment='top')

    ax_bar.set_ylim(0, 1)
    ax_bar.set_title('metrics')
    ax_bar.legend(loc='lower right')
    plt.suptitle(title)

    plt.tight_layout(h_pad=2)
    return


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
    ax_acc.plot(train_accuracies)
    ax_acc.plot(val_accuracies)

    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.plot(train_losses)
    ax_loss.plot(val_losses)

    fig.tight_layout()
