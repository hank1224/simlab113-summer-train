from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Config:
    POP_SIZE = 5000  # 種群大小
    GENERATIONS = 100  # 世代數
    MUTATION_DELTA = 0.5  # 變異幅度
    MUTATION_RATE = 0.6  # 變異率

    SELECTION_STRATEGY_TYPE = "roulette"  # "roulette", "tournament", "rank_based"
    TOURNAMENT_SIZE = 5  # 錦標賽選擇(tournament)的大小

    CURRENT_GENERATION = 0


"""
    定義和計算目標的函數。
"""


class ObjectiveFunction:
    @staticmethod
    def evaluate(x, y):
        return -np.cos(np.pi * y) - np.exp(-np.pi * (x - 0.5) ** 2) * np.sin(np.pi * x) ** 2


"""
    表示種群中的個體，包含 x 和 y 變量。
"""


class Individual:
    # 初始化個體，沒有提供 x 和 y 則隨機生成
    def __init__(self, x=None, y=None):
        if x is None or y is None:
            self.x = np.random.uniform(-1, 2)
            self.y = np.random.uniform(4, 6)
            while self.x + self.y < 5:
                self.x = np.random.uniform(-1, 2)
                self.y = np.random.uniform(4, 6)
        else:
            self.x = x
            self.y = y

    # 計算個體的適應度
    def fitness(self):
        return ObjectiveFunction.evaluate(self.x, self.y)

    # 均勻突變（Uniform Mutation）
    def mutate(self, mutation_rate=Config.MUTATION_RATE):
        if np.random.rand() < mutation_rate:  # 以 mutation_rate 的機率進行變異
            self.x += np.random.uniform(-Config.MUTATION_DELTA, Config.MUTATION_DELTA)  # 變異幅度
            self.y += np.random.uniform(-Config.MUTATION_DELTA, Config.MUTATION_DELTA)

            # 防止變異後超出範圍，限制在 [-1, 2] 和 [4, 6] 之間
            self.x = np.clip(self.x, -1, 2)
            self.y = np.clip(self.y, 4, 6)

            # 如果變異後不符合限制條件，
            if self.x + self.y < 5:
                # 隨機調整 x 或 y 以滿足 x + y >= 5
                if np.random.rand() < 0.5:
                    self.x = 5 - self.y
                    self.x = np.clip(self.x, -1, 2)  # 再次確認 x 在範圍內
                else:
                    self.y = 5 - self.x
                    self.y = np.clip(self.y, 4, 6)  # 再次確認 y 在範圍內

    # # 高斯突變（Gaussian Mutation）
    # def mutate(self, mutation_rate=Config.MUTATION_RATE, sigma=0.1):
    #     if np.random.rand() < mutation_rate:
    #         self.x += np.random.normal(0, sigma)
    #         self.x = np.clip(self.x, -1, 2)  # 確保 x 在範圍內
    #     if np.random.rand() < mutation_rate:
    #         self.y += np.random.normal(0, sigma)
    #         self.y = np.clip(self.y, 4, 6)  # 確保 y 在範圍內
    #     # 確保 x 和 y 的約束條件
    #     while self.x + self.y < 5:
    #         self.x = np.random.uniform(-1, 2)
    #         self.y = np.random.uniform(4, 6)

    # 非均勻突變（Non-uniform Mutation） 當前世代數問題！！！！
    # def mutate(self, mutation_rate=Config.MUTATION_RATE, current_generation=1, max_generations=Config.GENERATIONS):
    #     if np.random.rand() < mutation_rate:
    #         tau = (1 - current_generation / max_generations) ** 2
    #         self.x += np.random.uniform(-1, 1) * tau
    #         self.x = np.clip(self.x, -1, 2)  # 確保 x 在範圍內
    #     if np.random.rand() < mutation_rate:
    #         tau = (1 - current_generation / max_generations) ** 2
    #         self.y += np.random.uniform(-1, 1) * tau
    #         self.y = np.clip(self.y, 4, 6)  # 確保 y 在範圍內
    #     # 確保 x 和 y 的約束條件
    #     while self.x + self.y < 5:
    #         self.x = np.random.uniform(-1, 2)
    #         self.y = np.random.uniform(4, 6)


"""
    選擇策略的抽象基類，定義了選擇父母的方法。
"""


class SelectionStrategy(ABC):
    @abstractmethod
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        pass


class RouletteWheelSelection(SelectionStrategy):
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        fitness = np.array([individual.fitness() for individual in population])
        fitness -= fitness.min()
        if fitness.sum() == 0:
            fitness = np.ones_like(fitness)
        probabilities = fitness / fitness.sum()
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]


class TournamentSelection(SelectionStrategy):
    def __init__(self, tournament_size: int):
        self.tournament_size = tournament_size

    def select_parents(self, population: List[Individual]) -> List[Individual]:
        fitness = np.array([individual.fitness() for individual in population])
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), self.tournament_size)
            best = tournament[np.argmax(fitness[tournament])]
            selected.append(population[best])
        return selected


class RankBasedSelection(SelectionStrategy):
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        fitness = np.array([individual.fitness() for individual in population])
        ranks = np.argsort(np.argsort(fitness))
        probabilities = ranks / ranks.sum()
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]


"""
    管理種群，包括初始化、選擇、交叉和變異操作。
"""


class Population:
    def __init__(self, size, selection_strategy: SelectionStrategy):
        self.individuals = [Individual() for _ in range(size)]
        self.selection_strategy = selection_strategy

    # 計算種群中所有個體的適應度
    def evaluate(self):
        return [individual.fitness() for individual in self.individuals]

    # 選擇父母
    def select_parents(self):
        return self.selection_strategy.select_parents(self.individuals)  # 由選擇策略來選擇父母

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        x_child = alpha * parent1.x + (1 - alpha) * parent2.x
        y_child = alpha * parent1.y + (1 - alpha) * parent2.y
        return Individual(x_child, y_child)

    def generate_next_population(self, parents, mutation_rate):
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            next_population.extend([child1, child2])
        self.individuals = next_population

    def get_best_individual(self):
        return min(self.individuals, key=lambda individual: individual.fitness())


"""
    管理基因演算法的整個流程，包括種群的進化過程。
"""


class GeneticAlgorithm:
    def __init__(self, pop_size=Config.POP_SIZE, generations=Config.GENERATIONS, \
                 selection_strategy=Config.SELECTION_STRATEGY_TYPE, mutation_rate=Config.MUTATION_RATE):
        self.pop_size = pop_size
        self.generations = generations
        self.selection_strategy = selection_strategy
        self.mutation_rate = mutation_rate
        self.current_generation = 0

    def run(self):
        population = Population(self.pop_size, self.selection_strategy)
        global_best_individual = population.get_best_individual()
        for generation in range(self.generations):
            parents = population.select_parents()
            population.generate_next_population(parents, self.mutation_rate)
            best_individual = population.get_best_individual()
            if best_individual.fitness() < global_best_individual.fitness():
                global_best_individual = best_individual
            self.current_generation = generation

            print(f"Generation {generation + 1}:")
            print(
                f"  Current Best Individual: (x: {best_individual.x:.6f}, y: {best_individual.y:.6f}), Fitness = {best_individual.fitness():.6f}")
            print(
                f"  Global Best Individual:  (x: {global_best_individual.x:.6f}, y: {global_best_individual.y:.6f}), Fitness = {global_best_individual.fitness():.6f}")
            print("-" * 50)

        return global_best_individual


def get_selection_strategy(strategy_type, tournament_size=None):
    """
    根據給定的策略類型返回對應的選擇策略實例。

    :param strategy_type: 選擇策略的類型
    :param tournament_size: 錦標賽選擇策略的錦標賽大小
    :return: 選擇策略實例
    :raises ValueError: 當策略類型無效時拋出
    """
    strategies = {
        "roulette": RouletteWheelSelection,
        "tournament": lambda: TournamentSelection(tournament_size=tournament_size),
        "rank_based": RankBasedSelection
    }

    if strategy_type not in strategies:
        raise ValueError(f"Invalid selection strategy: {strategy_type}")

    return strategies[strategy_type]()


if __name__ == "__main__":
    try:
        selection_strategy = get_selection_strategy(
            Config.SELECTION_STRATEGY_TYPE,
            Config.TOURNAMENT_SIZE
        )
        ga = GeneticAlgorithm(selection_strategy=selection_strategy)
        best_solution = ga.run()
        print(f"最佳解: x = {best_solution.x}, y = {best_solution.y}, f(x, y) = {best_solution.fitness()}")

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Runtime Error: {e}")
