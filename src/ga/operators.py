import numpy as np
from random import randint, random


def tournament_selection(population, fitnesses, k=3):
    selected = np.random.choice(len(population), k, replace=False)
    best_individual = selected[0]
    best_fitness = fitnesses[selected[0]]

    for i in selected[1:]:
        if fitnesses[i] > best_fitness:
            best_fitness = fitnesses[i]
            best_individual = i

    return population[best_individual]


def crossover(parent1, parent2):
    crossover_point = randint(1, len(parent1) - 1)

    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2


def mutate(chromosome, mutation_rate=0.1):
    chromosome = np.array(chromosome)
    for i in range(len(chromosome)):
        if np.random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome
