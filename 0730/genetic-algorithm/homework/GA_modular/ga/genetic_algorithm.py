import matplotlib.pyplot as plt

from config import RuntimeState, GeneticAlgorithmConfig
from entities import Individual, Population


class GeneticAlgorithm:

    def __init__(self):
        self.GAconfig = GeneticAlgorithmConfig()
        self.individual = Individual()
        self.population = Population(self.GAconfig.POP_SIZE)
        self.runtime_stats = RuntimeState()
        self.fitness_history = []  # 用於存儲每次迭代的最佳適應度

    def __log_generation_stats(self, best_individual):
        print(f"Generation {self.runtime_stats.get_current_generation()}:")
        print(
            f"  Current Best Individual: (x: {best_individual.x:.6f}, "
            f"y: {best_individual.y:.6f}), "
            f"Fitness = {best_individual.fitness():.6f}")
        print(
            f"  Global Best Individual:  (x: {self.runtime_stats.get_global_best_individual().x:.6f}, "
            f"y: {self.runtime_stats.get_global_best_individual().y:.6f}),"
            f"Fitness = {self.runtime_stats.get_global_best_fitness():.6f}")
        print("-" * 50)

    def __print_final_result(self):
        print("Global Best Individual:")
        print(
            f"  (x: {self.runtime_stats.get_global_best_individual().x:.6f}, "
            f"y: {self.runtime_stats.get_global_best_individual().y:.6f}),"
            f"Fitness = {self.runtime_stats.get_global_best_fitness():.6f}")

    def __plot_fitness(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(GeneticAlgorithmConfig.GENERATIONS), self.fitness_history, marker='o', linestyle='-', color='b')
        plt.title('Fitness over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.show()

    def run(self):
        # 初始化全局最佳個體和適應度
        global_best_individual = self.population.get_best_individual()
        self.runtime_stats.set_global_best_individual(global_best_individual)
        self.runtime_stats.set_global_best_fitness(global_best_individual.fitness())

        # 開始迭代
        for generation in range(self.GAconfig.get_generations()):
            # 選擇父母
            parents = self.population.select_parents()
            # 生成下一代
            self.population.generate_next_population(parents)
            # 計算當前最佳個體
            best_individual = self.population.get_best_individual()

            # 更新全局最佳個體和適應度
            if best_individual.fitness() < self.runtime_stats.get_global_best_fitness():
                self.runtime_stats.set_global_best_individual(best_individual)
                self.runtime_stats.set_global_best_fitness(best_individual.fitness())

            # 記錄當前最佳適應度
            self.fitness_history.append(best_individual.fitness())

            # 輸出當前世代的統計資料
            self.__log_generation_stats(best_individual)
            # 更新計數
            self.runtime_stats.increment_current_generation()

        # 輸出最終結果
        self.__print_final_result()
        # 繪製適應度變化圖表
        self.__plot_fitness()
