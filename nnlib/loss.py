from abc import abstractmethod
from typing import Tuple

import numpy as np

from nnlib.module import Module


class Loss(Module):
    """
    Ошибка
    """

    last_input: Tuple[np.ndarray, np.ndarray]

    @abstractmethod
    def loss_func(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ) -> np.ndarray:
        """
        Функция ошибки

        Args:
            y_true: np.ndarray
            y_pred: np.ndarray

        Returns:
            np.ndarray
        """
        pass

    @abstractmethod
    def loss_func_prime(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ) -> np.ndarray:
        """
        Производная ошибки

        Args:
            y_true: np.ndarray
            y_pred: np.ndarray

        Returns:
            np.ndarray
        """
        pass

    def forward(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ) -> np.ndarray:
        self.last_input = (y_true, y_pred)
        return self.loss_func(*self.last_input)

    def backward(
            self,
            loss: np.ndarray,
            learning_rate: float,
            ) -> np.ndarray:
        return self.loss_func_prime(*self.last_input) * loss


class MseLoss(Loss):
    def loss_func(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ) -> np.ndarray:
        return np.mean((y_true - y_pred) ** 2)

    def loss_func_prime(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            ) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]
