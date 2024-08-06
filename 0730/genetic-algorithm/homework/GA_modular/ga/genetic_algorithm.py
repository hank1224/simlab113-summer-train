from config import settings, RuntimeState
from entities import Individual, Population


class GeneticAlgorithm:

    def __init__(self):
        self.GAconfig = settings.GeneticAlgorithmConfig()
        self.individual = Individual()
        self.population = Population(self.GAconfig.POP_SIZE)
        self.runtime_stats = RuntimeState()

    def __log_generation_stats(self, generation, best_individual):
        print(f"Generation {generation}:")
        print(
            f"  Current Best Individual: (x: {best_individual.x:.6f}, "
            f"y: {best_individual.y:.6f}), "
            f"Fitness = {best_individual.fitness():.6f}")
        print(
            f"  Global Best Individual:  (x: {self.runtime_stats.get_global_best_individual().x:.6f}, "
            f"y: {self.runtime_stats.get_global_best_individual().y:.6f}),"
            f"Fitness = {self.runtime_stats.get_global_best_fitness():.6f}")
        print("-" * 50)

    def run(self):
        # 初始化全局最佳個體和適應度
        global_best_individual = self.population.get_best_individual()
        self.runtime_stats.set_global_best_individual(global_best_individual)
        self.runtime_stats.set_global_best_fitness(global_best_individual.fitness())

        for generation in range(self.GAconfig.get_generations()):

            parents = self.population.select_parents()
            self.population.generate_next_population(parents)
            best_individual = self.population.get_best_individual()

            if best_individual.fitness() < self.runtime_stats.get_global_best_fitness():
                self.runtime_stats.set_global_best_individual(best_individual)
                self.runtime_stats.set_global_best_fitness(best_individual.fitness())

            self.__log_generation_stats(generation, best_individual)

        print("Final Global Best Individual:")
        print(
            f"  (x: {self.runtime_stats.get_global_best_individual().x:.6f}, "
            f"y: {self.runtime_stats.get_global_best_individual().y:.6f}),"
            f"Fitness = {self.runtime_stats.get_global_best_fitness():.6f}")
