from typing import List

import numpy as np

from nnlib.loss import Loss
from nnlib.module import Module


class Sequential:
    def __init__(
            self,
            layers: List[Module],
            loss: Loss,
            learning_rate: float,
            ):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.loss_values = []

    def forward(self, x: np.ndarray):
        # x: [batch_size, *dims]

        output = x
        for layer in self.layers:
            output = layer(output)

        return output

    def backward(
            self,
            y_train: np.ndarray,
            x_train: np.ndarray,
            ) -> np.ndarray:
        loss_value = self.loss(y_train, x_train).sum()
        grad = self.loss.backward(loss_value, self.learning_rate)
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, self.learning_rate)
        return loss_value

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            num_epochs: int
            ) -> None:

        for epoch in range(num_epochs):
            pred = self.forward(x_train)
            loss_value = self.backward(y_train, pred)
            self.loss_values.append(loss_value)

