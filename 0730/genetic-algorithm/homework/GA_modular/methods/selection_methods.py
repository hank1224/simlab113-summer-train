from typing import List

import numpy as np

from config.settings import SelectionConfig
from entities import Individual


def rank_base_selection(population: List[Individual]) -> List[Individual]:
    fitness = np.array([individual.fitness() for individual in population])
    ranks = np.argsort(np.argsort(fitness))
    probabilities = ranks / ranks.sum()
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return [population[i] for i in selected_indices]


def roulette_wheel_selection(population: List[Individual]) -> List[Individual]:
    fitness = np.array([individual.fitness() for individual in population])
    fitness -= fitness.min()
    if fitness.sum() == 0:
        fitness = np.ones_like(fitness)
    probabilities = fitness / fitness.sum()
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return [population[i] for i in selected_indices]


def tournament_selection(population: List[Individual]) -> List[Individual]:
    fitness = np.array([individual.fitness() for individual in population])
    selected_indices = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), size=SelectionConfig.TOURNAMENT_SIZE)
        best = tournament[np.argmax(fitness[tournament])]
        selected_indices.append(best)
    return [population[i] for i in selected_indices]
