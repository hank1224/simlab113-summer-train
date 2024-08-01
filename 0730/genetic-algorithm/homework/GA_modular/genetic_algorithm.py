from config.RuntimeState import RuntimeState
from config.Config import *
from population.Population import Population

print("Initializing runtime statistics and population...")
runtime_stats = RuntimeState()
population = Population(GeneticAlgorithmConfig.POP_SIZE)

# 初始化全局最佳個體和適應度
global_best_individual = population.get_best_individual()
runtime_stats.set_global_best_individual(global_best_individual)
runtime_stats.set_global_best_fitness(global_best_individual.fitness())

for generation in range(GeneticAlgorithmConfig.GENERATIONS):  # 假設有一個配置類來存儲世代數
    runtime_stats.increment_current_generation()

    parents = population.select_parents()
    population.generate_next_population(parents)
    best_individual = population.get_best_individual()

    if best_individual.fitness() < runtime_stats.get_global_best_fitness():
        runtime_stats.set_global_best_individual(best_individual)
        runtime_stats.set_global_best_fitness(best_individual.fitness())

    print(f"Generation {runtime_stats.get_current_generation()}:")
    print(
        f"  Current Best Individual: (x: {best_individual.x:.6f}, "
        f"y: {best_individual.y:.6f}), "
        f"Fitness = {best_individual.fitness():.6f}")
    print(
        f"  Global Best Individual:  (x: {runtime_stats.get_global_best_individual().x:.6f}, "
        f"y: {runtime_stats.get_global_best_individual().y:.6f}),"
        f"Fitness = {runtime_stats.get_global_best_fitness():.6f}")
    print("-" * 50)

print("Final Global Best Individual:")
print(
    f"  (x: {runtime_stats.get_global_best_individual().x:.6f}, "
    f"y: {runtime_stats.get_global_best_individual().y:.6f}),"
    f"Fitness = {runtime_stats.get_global_best_fitness():.6f}")
