import numpy as np

from nnlib import nn, module, loss


if __name__ == '__main__':

    x_train = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ])
    y_train = np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    model = nn.Sequential(
        module.Linear(2, 3),
        module.Tanh(),
        module.Linear(3, 1),
        module.Tanh(),
        loss=loss.MseLoss(),
        learning_rate=0.1,
    )

    model.fit(x_train, y_train, 1000)

    print_freq = 100
    for epoch, val in enumerate(model.loss_values, start=1):
        if epoch % print_freq == 0:
            print(f'Epoch {epoch:6}, loss {val:10.4f}')

