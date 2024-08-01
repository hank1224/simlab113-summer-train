class RuntimeState:
    def __init__(self):
        self.current_generation = 0
        self.global_best_fitness = float('-inf')
        self.global_best_individual = None

    # getters and setters
    def get_current_generation(self):
        return self.current_generation

    def increment_current_generation(self):
        self.current_generation += 1

    def get_global_best_fitness(self):
        return self.global_best_fitness

    def set_global_best_fitness(self, global_best_fitness):
        self.global_best_fitness = global_best_fitness

    def get_global_best_individual(self):
        return self.global_best_individual

    def set_global_best_individual(self, global_best_individual):
        self.global_best_individual = global_best_individual
