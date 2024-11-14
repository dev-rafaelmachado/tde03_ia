from typing import List, Tuple
import numpy as np
from .fitness import fitness
from .operators import tournament_selection, crossover, mutate
import time
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier


def genetic_algorithm(
    X_train,
    y_train,
    X_test,
    y_test,
    num_features,
    population_size=20,
    generations=100,
    mutation_rate=0.01,
    verbose=True,
) -> Tuple[float, List[int], float, float]:
    start_time_feature = time.time()

    population = np.random.randint(2, size=(population_size, num_features))

    with tqdm(
        total=generations,
        desc="Generation Progress",
        bar_format="{l_bar}{bar}| {n_fmt}% Complete",
        position=0,
    ) as progress_bar:
        for generation in range(generations):
            current_best_fitness = 0
            best_individual = []

            with tqdm(
                total=population_size,
                desc="Evaluating Population",
                leave=False,
                disable=not verbose,
                position=1,
            ) as inner_pbar:
                for individual in range(population_size):
                    fitness_value = fitness(
                        population[individual], X_train, y_train, X_test, y_test
                    )

                    if fitness_value > current_best_fitness:
                        current_best_fitness = fitness_value
                        best_individual = population[individual]

                    if verbose:
                        inner_pbar.set_postfix(
                            {
                                "Current Best Fitness": f"{current_best_fitness:.4f}",
                                "Individual": individual + 1,
                            }
                        )
                    inner_pbar.update(1)

            new_population = []
            while len(new_population) < population_size:
                parent1 = tournament_selection(
                    population,
                    [
                        fitness(individual, X_train, y_train, X_test, y_test)
                        for individual in population
                    ],
                )
                parent2 = tournament_selection(
                    population,
                    [
                        fitness(individual, X_train, y_train, X_test, y_test)
                        for individual in population
                    ],
                )

                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)

                new_population.append(child1)
                new_population.append(child2)

            population = np.array(new_population[:population_size])

            progress_bar.n = generation + 1
            progress_bar.refresh()

    selected_features = [i for i, gene in enumerate(best_individual) if gene == 1]
    feature_search_time = time.time() - start_time_feature

    start_time_model = time.time()
    model = DecisionTreeClassifier()
    model.fit(X_train.iloc[:, selected_features], y_train)
    training_time = time.time() - start_time_model

    return (
        float(current_best_fitness),
        selected_features,
        feature_search_time,
        training_time,
    )
