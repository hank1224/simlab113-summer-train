import numpy as np


class ObjectiveFunction:
    @staticmethod
    def evaluate(x, y):
        return -np.cos(np.pi * y) - np.exp(-np.pi * (x - 0.5) ** 2) * np.sin(np.pi * x) ** 2


class Constraint:
    X_MIN = -1
    X_MAX = 2
    Y_MIN = 4
    Y_MAX = 6
    CONSTRAINT = lambda x, y: x + y >= 5
