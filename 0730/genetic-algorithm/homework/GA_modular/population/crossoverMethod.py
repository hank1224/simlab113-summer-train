import numpy as np
from individual.Individual import Individual
from config.Config import CrossoverConfig


def uniform_crossover(parent1, parent2):
    x_child = parent1.x if np.random.rand() < 0.5 else parent2.x
    y_child = parent1.y if np.random.rand() < 0.5 else parent2.y
    return Individual(x_child, y_child)


def linear_interpolation_crossover(parent1, parent2):
    alpha = np.random.rand()
    x_child = alpha * parent1.x + (1 - alpha) * parent2.x
    y_child = alpha * parent1.y + (1 - alpha) * parent2.y
    return Individual(x_child, y_child)


def blx_alpha_crossover(parent1, parent2, alpha=CrossoverConfig.BLX_ALPHA):
    x_min = min(parent1.x, parent2.x)
    x_max = max(parent1.x, parent2.x)
    y_min = min(parent1.y, parent2.y)
    y_max = max(parent1.y, parent2.y)

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_child = np.random.uniform(x_min - alpha * x_range, x_max + alpha * x_range)
    y_child = np.random.uniform(y_min - alpha * y_range, y_max + alpha * y_range)

    return Individual(x_child, y_child)
