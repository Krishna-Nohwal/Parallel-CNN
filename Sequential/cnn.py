"""NumPy implementation of the sequential MNIST CNN."""

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


DT = 0.1


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for the range used by this network."""
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


class SequentialCNN:
    """CNN with the same layer sizes and update rules as the C++ baseline."""

    def __init__(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.conv_weight = rng.uniform(-0.5, 0.5, (6, 5, 5)).astype(np.float32)
        self.conv_bias = rng.uniform(-0.5, 0.5, 6).astype(np.float32)
        self.pool_weight = rng.uniform(-0.5, 0.5, (4, 4)).astype(np.float32)
        self.pool_bias = rng.uniform(-0.5, 0.5, 1).astype(np.float32)
        self.fc_weight = rng.uniform(-0.5, 0.5, (10, 6, 6, 6)).astype(np.float32)
        self.fc_bias = rng.uniform(-0.5, 0.5, 10).astype(np.float32)
        self.times = {"convolution": 0.0, "pooling": 0.0, "fully connected": 0.0, "gradients": 0.0}

    def forward(self, image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        conv_windows = sliding_window_view(image, (5, 5))
        start = time.perf_counter()
        conv_preact = np.einsum("xyij,mij->mxy", conv_windows, self.conv_weight, optimize=True)
        conv_preact += self.conv_bias[:, None, None]
        conv_output = sigmoid(conv_preact)
        self.times["convolution"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        pool_windows = conv_output.reshape(6, 6, 4, 6, 4)
        pool_preact = np.einsum("mxiyj,ij->mxy", pool_windows, self.pool_weight, optimize=True)
        pool_preact += self.pool_bias[0]
        pool_output = sigmoid(pool_preact)
        self.times["pooling"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        fc_preact = np.einsum("omxy,mxy->o", self.fc_weight, pool_output, optimize=True)
        fc_preact += self.fc_bias
        fc_output = sigmoid(fc_preact)
        self.times["fully connected"] += (time.perf_counter() - start) * 1000
        return fc_output, (
            conv_windows,
            conv_output,
            pool_windows,
            pool_output,
            fc_output,
        )

    def backprop(self, label: int, cache: tuple[np.ndarray, ...]) -> None:
        conv_windows, conv_output, pool_windows, pool_output, fc_output = cache

        start = time.perf_counter()
        fc_delta = -fc_output.copy()
        fc_delta[label] += 1.0
        fc_dweight = fc_delta[:, None, None, None] * pool_output[None, :, :, :]
        self.fc_bias += DT * fc_delta

        pool_output_delta = np.einsum("omxy,o->mxy", self.fc_weight, fc_delta, optimize=True)
        pool_delta = pool_output_delta * pool_output * (1.0 - pool_output)

        pool_dweight = np.einsum("mxy,mxiyj->ij", pool_delta, pool_windows, optimize=True)
        self.pool_bias[0] += DT * float(pool_delta.mean())

        conv_output_delta = np.zeros_like(conv_output)
        conv_output_delta.reshape(6, 6, 4, 6, 4)[:] = (
            self.pool_weight[None, None, :, None, :]
            * pool_delta.reshape(6, 6, 1, 6, 1)
        )
        conv_delta = conv_output_delta * conv_output * (1.0 - conv_output)

        conv_dweight = np.einsum("mxy,xyij->mij", conv_delta, conv_windows, optimize=True)
        conv_dweight /= 24.0 * 24.0
        self.conv_bias += DT * conv_delta.mean(axis=(1, 2))

        self.fc_weight += DT * fc_dweight
        self.pool_weight += DT * pool_dweight
        self.conv_weight += DT * conv_dweight
        self.times["gradients"] += (time.perf_counter() - start) * 1000

    def classify(self, image: np.ndarray) -> int:
        output, _ = self.forward(image)
        return int(np.argmax(output))
