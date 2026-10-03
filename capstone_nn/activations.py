import numpy as np


class ReLU:

    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x

        return np.maximum(
                0,
                x,
            )

    def backward(self, grad_output):

        if self.input is None:
            raise RuntimeError(
                    "forward() must be called before backward()."
                )

        grad_input = grad_output.copy()

        grad_input[
                self.input <= 0
            ] = 0

        return grad_input


