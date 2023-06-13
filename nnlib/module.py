from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Module(ABC):
    last_input: np.ndarray

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dEdY: np.ndarray, learning_rate: float) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ActivationLayer(Module):
    """
    Слой активации
    """

    @abstractmethod
    def activation(self, x: np.ndarray) -> np.ndarray:
        """
        Функция активации

        Args:
            x: np.ndarray

        Returns:
            np.ndarray
        """
        pass

    @abstractmethod
    def activation_prime(self, x: np.ndarray) -> np.ndarray:
        """
        Производная активации

        Args:
            x: np.ndarray

        Returns:
            np.ndarray
        """
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return self.activation(x)

    def backward(self, dEdY: np.ndarray, learning_rate: float) -> np.ndarray:
        return self.activation_prime(self.last_input).mean(axis=0) * dEdY


class ReLU(ActivationLayer):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def activation_prime(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Tanh(ActivationLayer):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def activation_prime(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2


class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        self.bias = np.random.uniform(-0.5, 0.5, (1, output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return x @ self.weights + self.bias

    def backward(self, dEdY: np.ndarray, learning_rate: float) -> np.ndarray:
        # E - error
        # X - input to this layer
        # Y - output of this layer

        # Gradients
        dEdX = dEdY @ self.weights.T
        dEdW = self.last_input.T.mean(axis=-1)[..., None] @ dEdY[None, ...]
        dEdB = dEdY

        # Weights update
        self.weights -= learning_rate * dEdW
        self.bias -= learning_rate * dEdB

        return dEdX
