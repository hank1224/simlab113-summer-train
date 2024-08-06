import numpy as np
from config.problem import Constraint
from config.settings import GeneticAlgorithmConfig, MutationConfig


def uniform_mutation(individual):
    if np.random.rand() < MutationConfig.MUTATION_RATE:  # 以 mutation_rate 的機率進行變異
        individual.x += np.random.uniform(-MutationConfig.MUTATION_DELTA, MutationConfig.MUTATION_DELTA)  # 變異幅度
        individual.y += np.random.uniform(-MutationConfig.MUTATION_DELTA, MutationConfig.MUTATION_DELTA)

        # 防止變異後超出範圍，限制在 Constraint 的範圍內
        individual.x = np.clip(individual.x, Constraint.X_MIN, Constraint.X_MAX)
        individual.y = np.clip(individual.y, Constraint.Y_MIN, Constraint.Y_MAX)

        # 如果變異後不符合限制條件，
        if not Constraint.CONSTRAINT(individual.x, individual.y):
            # 隨機調整 x 或 y 以滿足約束條件
            if np.random.rand() < 0.5:
                individual.x = 5 - individual.y
                individual.x = np.clip(individual.x, Constraint.X_MIN, Constraint.X_MAX)  # 再次確認 x 在範圍內
            else:
                individual.y = 5 - individual.x
                individual.y = np.clip(individual.y, Constraint.Y_MIN, Constraint.Y_MAX)  # 再次確認 y 在範圍內

    return individual.x, individual.y


def gaussian_mutate(individual, sigma=0.1):
    if np.random.rand() < MutationConfig.MUTATION_RATE:
        individual.x += np.random.normal(0, sigma)
        individual.x = np.clip(individual.x, Constraint.X_MIN, Constraint.X_MAX)  # 確保 x 在範圍內
    if np.random.rand() < MutationConfig.MUTATION_RATE:
        individual.y += np.random.normal(0, sigma)
        individual.y = np.clip(individual.y, Constraint.Y_MIN, Constraint.Y_MAX)  # 確保 y 在範圍內
    # 確保 x 和 y 的約束條件
    while not Constraint.CONSTRAINT(individual.x, individual.y):
        individual.x = np.random.uniform(Constraint.X_MIN, Constraint.X_MAX)
        individual.y = np.random.uniform(Constraint.Y_MIN, Constraint.Y_MAX)

    return individual.x, individual.y


def non_uniform_mutation(individual, current_generation, max_generations=GeneticAlgorithmConfig.GENERATIONS):
    if np.random.rand() < MutationConfig.MUTATION_RATE:
        tau = (1 - current_generation / max_generations) ** 2
        individual.x += np.random.uniform(-1, 1) * tau
        individual.x = np.clip(individual.x, Constraint.X_MIN, Constraint.X_MAX)  # 確保 x 在範圍內
    if np.random.rand() < MutationConfig.MUTATION_RATE:
        tau = (1 - current_generation / max_generations) ** 2
        individual.y += np.random.uniform(-1, 1) * tau
        individual.y = np.clip(individual.y, Constraint.Y_MIN, Constraint.Y_MAX)  # 確保 y 在範圍內
    # 確保 x 和 y 的約束條件
    while not Constraint.CONSTRAINT(individual.x, individual.y):
        individual.x = np.random.uniform(Constraint.X_MIN, Constraint.X_MAX)
        individual.y = np.random.uniform(Constraint.Y_MIN, Constraint.Y_MAX)

    return individual.x, individual.y
