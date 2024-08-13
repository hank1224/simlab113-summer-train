import numpy as np

from config.problem import ObjectiveFunction, Constraint
from config.settings import MutationConfig
from methods import mutation_methods


class Individual:
    # 初始化個體，沒有提供 x 和 y 則隨機生成
    def __init__(self, x=None, y=None):
        if x is None or y is None:
            self.x = np.random.uniform(Constraint.X_MIN, Constraint.X_MAX)
            self.y = np.random.uniform(Constraint.Y_MIN, Constraint.Y_MAX)
            while not Constraint.CONSTRAINT(self.x, self.y):
                self.x = np.random.uniform(Constraint.X_MIN, Constraint.X_MAX)
                self.y = np.random.uniform(Constraint.Y_MIN, Constraint.Y_MAX)
        else:
            self.x = x
            self.y = y

    # 計算個體的適應度
    def fitness(self):
        return ObjectiveFunction.evaluate(self.x, self.y)

    def mutate(self, mutation_type=MutationConfig.MUTATION_TYPE):
        match mutation_type:
            case "uniform":
                mutation_methods.uniform_mutation(self)
            case "gaussian":
                mutation_methods.gaussian_mutation(self)
            case _:
                raise ValueError(f"Invalid mutation type: {mutation_type}")
