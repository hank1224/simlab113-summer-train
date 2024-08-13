from config.settings import SelectionConfig, CrossoverConfig
from entities import Individual
from methods import selection_methods, crossover_methods


class Population:
    def __init__(self, size):
        self.individuals = [Individual() for _ in range(size)]

    # 計算種群中所有個體的適應度
    def evaluate(self):
        return [individual.fitness() for individual in self.individuals]

    def select_parents(self, selection_strategy=SelectionConfig.SELECTION_STRATEGY_TYPE):
        match selection_strategy:
            case "roulette":
                return selection_methods.roulette_wheel_selection(self.individuals)
            case "tournament":
                return selection_methods.tournament_selection(self.individuals)
            case "rank_based":
                return selection_methods.rank_base_selection(self.individuals)
            case _:
                raise ValueError(f"Invalid selection strategy: {selection_strategy}")

    def crossover(self, parent1, parent2):
        match CrossoverConfig.CROSSOVER_TYPE:
            case "uniform":
                return crossover_methods.uniform_crossover(parent1, parent2)
            case "blx_alpha":
                return crossover_methods.blx_alpha_crossover(parent1, parent2)
            case "linear_interpolation":
                return crossover_methods.linear_interpolation_crossover(parent1, parent2)
            case _:
                raise ValueError(f"Invalid crossover type: {CrossoverConfig.CROSSOVER_TYPE}")

    def generate_next_population(self, parents):
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            child1.mutate()
            child2.mutate()
            next_population.extend([child1, child2])
        self.individuals = next_population

    def get_best_individual(self):
        return min(self.individuals, key=lambda individual: individual.fitness())
