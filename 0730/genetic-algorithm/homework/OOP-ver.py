import numpy as np

POP_SIZE = 500 # 種群大小
GENERATIONS = 10 # 世代數
MUTATION_DELTA = 0.5 # 變異幅度
MUTATION_RATE = 0.6 # 變異率

"""
    定義和計算目標的函數。
"""
class ObjectiveFunction:
    @staticmethod
    def evaluate(x, y):
        return -np.cos(np.pi * y) - np.exp(-np.pi * (x - 0.5)**2) * np.sin(np.pi * x)**2

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

    # 根據變異率對個體進行變
    def mutate(self, mutation_rate=MUTATION_RATE):
        if np.random.rand() < mutation_rate:
            self.x += np.random.uniform(-MUTATION_DELTA, MUTATION_DELTA)
            self.y += np.random.uniform(-MUTATION_DELTA, MUTATION_DELTA)
            self.x = np.clip(self.x, -1, 2)
            self.y = np.clip(self.y, 4, 6)
            if self.x + self.y < 5:
                self.y = 5 - self.x

"""
    管理種群，包括初始化、選擇、交叉和變異操作。
"""
class Population:
    def __init__(self, size):
        self.individuals = [Individual() for _ in range(size)]

    # 計算種群中所有個體的適應度
    def evaluate(self):
        return [individual.fitness() for individual in self.individuals]

    # 根據適應度選擇父母
    def select_parents(self):
        fitness = np.array(self.evaluate())
        fitness -= fitness.min()
        if fitness.sum() == 0:
            fitness = np.ones_like(fitness)
        probabilities = fitness / fitness.sum()
        selected_indices = np.random.choice(len(self.individuals), size=len(self.individuals), p=probabilities)
        selected = [self.individuals[i] for i in selected_indices]
        return selected

    # 生成子代
    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        x_child = alpha * parent1.x + (1 - alpha) * parent2.x
        y_child = alpha * parent1.y + (1 - alpha) * parent2.y
        return Individual(x_child, y_child)

    # 生成下一代種群
    def generate_next_population(self, parents, mutation_rate):
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            next_population.extend([child1, child2])
        self.individuals = next_population

    # 獲取種群中適應度最好的個體
    def get_best_individual(self):
        return min(self.individuals, key=lambda individual: individual.fitness())

"""
    管理基因演算法的整個流程，包括種群的進化過程。
"""
class GeneticAlgorithm:
    def __init__(self, pop_size=POP_SIZE, generations=GENERATIONS, mutation_rate=MUTATION_RATE):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def run(self):
        population = Population(self.pop_size)
        global_best_individual = population.get_best_individual()
        for generation in range(self.generations):
            fitness = population.evaluate()
            parents = population.select_parents()
            population.generate_next_population(parents, self.mutation_rate)
            best_individual = population.get_best_individual()
            if best_individual.fitness() < global_best_individual.fitness():
                global_best_individual = best_individual

            print(f"Generation {generation+1}:")
            print(f"  Current Best Individual: (x: {best_individual.x:.6f}, y: {best_individual.y:.6f}), Fitness = {best_individual.fitness():.6f}")
            print(f"  Global Best Individual:  (x: {global_best_individual.x:.6f}, y: {global_best_individual.y:.6f}), Fitness = {global_best_individual.fitness():.6f}")
            print("-" * 50)

        return global_best_individual


ga = GeneticAlgorithm()
best_solution = ga.run()
print(f"最佳解: x = {best_solution.x}, y = {best_solution.y}, f(x, y) = {best_solution.fitness()}")
