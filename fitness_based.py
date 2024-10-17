"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This file implements the fitness sharing evolutionary algorithm.
This is a modified version of the EA1 algorithm that includes fitness sharing
to encourage diversity in the population. The EA1 algorithm is modified to
include a sharing function that calculates the portion of fitness that one
should get based on the distance between the individuals. This sharing function
is then applied directly into the fitness to encourage diversity in the
population. The EA1 algorithm is then run with the fitness sharing mechanism
to compare the results with the standard EA1 algorithm.
"""

# imports framework
import time
import csv
import itertools

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# imports file utils and functions
from file_utils import save_best_solution, save_fitness_stats_to_csv

# standard offensive strategy first
gamma = 0.9
beta = 0.1
EA_NAME = 'EA1'


# Custom environment class to change the fitness function
class CustomEnvironment(Environment):
    def fitness_single(self):
        time_score = 0
        kill_bonus = 0

        if self.get_playerlife() <= 0:
            time_score = np.log(self.get_time())
        else:
            time_score = -np.log(self.get_time())

        if self.get_enemylife() <= 0:
            kill_bonus = 100

        return gamma * (100 - self.get_enemylife()) + beta * self.get_playerlife() + time_score + kill_bonus


# EA class for running the evolutionary algorithm
class EA:
    def __init__(self, population_size, n_vars, upper_bound, lower_bound,
                 crossover_rate, mutation_rate, mutation_std, tournament_size,
                 alpha, env, no_generations, enemies):
        self.population_size = population_size
        self.n_vars = n_vars
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.no_generations = no_generations
        self.enemies = enemies  # now we have multiple enemies

        # Update environment with the specified enemy
        self.env = env
        self.env.update_parameter('enemies', enemies)

        self.population = self.initialize_population()
        self.fitness_population = self.evaluate_population()

        # Initialise the best solution and fitness to negative infinity
        self.best_solution = None
        self.best_solution_fitness = float('-inf')

        # Initialize list to store fitness statistics and variety per generation
        self.fitness_stats = []

        # Ensure the data folder exists to store CSV files
        if not os.path.exists("testdata"):
            os.makedirs("testdata")

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.n_vars))

    def clamp(self, individual):
        return np.clip(individual, self.lower_bound, self.upper_bound)

    def tournament_selection(self, fitness_population):
        selected_indices = np.random.choice(self.population.shape[0], self.tournament_size, replace=False)
        selected_fitness = fitness_population[selected_indices]
        best_index = np.argmax(selected_fitness)
        return self.population[selected_indices[best_index]]

    def crossover_single(self, parent1, parent2):
        offspring = np.zeros_like(parent1)

        # Perform blend crossover for each gene
        for i in range(len(parent1)):
            if np.random.uniform(0, 1) > self.crossover_rate:
                min_val = min(parent1[i], parent2[i])
                max_val = max(parent1[i], parent2[i])
                range_val = max_val - min_val
                offspring[i] = np.random.uniform(min_val - self.alpha * range_val, max_val + self.alpha * range_val)
            else:
                offspring[i] = parent1[i]

        return self.clamp(offspring)

    def mutate_individual(self, individual):
        for i in range(len(individual)):
            individual[i] += np.random.normal(0, self.mutation_std)

        return self.clamp(individual)

    def evaluate_population(self):
        return np.array([self.simulation(individual) for individual in self.population])

    # runs simulation
    def simulation(self, x):
        f, p, e, t = self.env.play(pcont=x)
        return f

    # calculates the euclidean distance between two individuals
    def distance(self, one, two):
        return np.linalg.norm(one - two)

    # Calculate the average variety (pairwise distance) in the population
    def calculate_diversity(self):
        total_distance = 0
        num_comparisons = 0

        # Loop through all pairs of individuals in the population
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):  # Avoid double counting
                total_distance += self.distance(self.population[i], self.population[j])
                num_comparisons += 1

        # Return the average distance (variety)
        if num_comparisons > 0:
            return total_distance / num_comparisons
        else:
            return 0

    # Store fitness statistics and variety for each generation
    def store_fitness_stats(self, generation, max_fitness, mean_fitness, std_fitness, variety):
        self.fitness_stats.append([generation, max_fitness, mean_fitness, std_fitness, variety])

    # The main evolutionary algorithm loop is now inside the run method
    def run(self):
        try:
            # Go for the set amount of generations
            for generation in range(self.no_generations):

                # get the fitness of the population
                fitness_population = self.fitness_population
                next_generation = []

                # Make new offspring until we have a new full generation
                for i in range(self.population_size):
                    # Select two parents using tournament selection
                    p1 = self.tournament_selection(fitness_population)
                    p2 = self.tournament_selection(fitness_population)
                    # ensure different p1 and p2
                    while np.array_equal(p1, p2):
                        p2 = self.tournament_selection(fitness_population)

                    # Perform crossover to produce offspring
                    offspring = self.crossover_single(p1, p2)

                    # Mutate offspring
                    if np.random.uniform(0, 1) < self.mutation_rate:
                        offspring = self.mutate_individual(offspring)

                    next_generation.append(offspring)

                # get the evaluation of the new generation
                next_generation_fitness = np.array([self.simulation(individual) for individual in next_generation])

                # stack both the generations pick the population size best individuals
                self.population = np.vstack((self.population, next_generation))
                self.fitness_population = np.hstack((self.fitness_population, next_generation_fitness))

                # pick the population size best individuals
                indices = np.argsort(self.fitness_population)[-self.population_size:]
                self.population = self.population[indices]
                self.fitness_population = self.fitness_population[indices]

                # Calculate fitness statistics
                generation_max_fitness = np.max(fitness_population)
                generation_mean_fitness = np.mean(fitness_population)
                generation_std_fitness = np.std(fitness_population)
                generation_variety = self.calculate_diversity()

                # Store fitness statistics and diversity for this generation
                self.store_fitness_stats(generation + 1, generation_max_fitness,
                                        generation_mean_fitness, generation_std_fitness,
                                        generation_variety)

                # Log the best fitness for monitoring
                if generation_max_fitness > self.best_solution_fitness:
                    self.best_solution = self.population[np.argmax(fitness_population)]
                    self.best_solution_fitness = generation_max_fitness

                print(f"{generation + 1}/{self.no_generations}, "
                    f"max: {round(generation_max_fitness, 1)}, "
                    f"mean: {round(generation_mean_fitness, 1)}, "
                    f"std: {round(generation_std_fitness, 1)}, "
                    f"diversity: {round(generation_variety, 1)}")

        except (KeyboardInterrupt, Exception) as e:
            print(f"\nInterrupted by {type(e).__name__}: {str(e)}")
            print("Saving the best solution found so far...")

        finally:
            # Save the best solution to a file using file_utils.py
            save_best_solution(self.best_solution, self.best_solution_fitness, self.enemies, EA_NAME)

            # Save fitness statistics to CSV using file_utils.py
            save_fitness_stats_to_csv(self.fitness_stats, self.enemies, EA_NAME)

            print("Best solution and fitness statistics saved.")
# checks at all the currnet runs and returns the remaining runs in a list, per
# run of a combination we add that combination to the list
def get_remaining_runs(folder_path, NUMBER_OF_RUNS):
    # Generate all combinations of 3 enemies out of 8
    enemy_combinations = list(itertools.combinations(range(1, 9), 3))

    folder_path = f'testdata/{EA_NAME}/'
    files_per_subfolder = {}

    # Loop over all the subdirectories in the main folder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):

            # Count the number of files in the subfolder and store
            num_files = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
            files_per_subfolder[tuple([int(el) for el in subfolder])] = num_files

    # get the complement of the already existing runs:
    remaining_runs = []
    for combination in enemy_combinations:
        if combination not in files_per_subfolder.keys():
            for i in range(NUMBER_OF_RUNS):
                remaining_runs.append(combination)
        else:
            if files_per_subfolder[combination] < NUMBER_OF_RUNS:
                for i in range(NUMBER_OF_RUNS - files_per_subfolder[combination]):
                    remaining_runs.append(combination)

    return remaining_runs

def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # Initializes simulation in individual evolution mode, for single static enemy.
    env = CustomEnvironment(experiment_name=experiment_name,
                            enemies=[1],
                            multiplemode="yes",
                            playermode="ai",
                            player_controller=player_controller(n_hidden_neurons),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
    n_input_nodes = env.get_num_sensors()

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # EA configuration
    population_size = 80
    # population_size = 10  # for now for testing
    no_generations = 30
    upper_bound = 1
    lower_bound = -1
    crossover_rate = 0.9
    alpha = 0.75
    mutation_rate = 0.22
    mutation_std = 0.45
    tournament_size = 7

    # enemies = [1]

    # test groups
    enemy_groups = [[1, 2, 3, 5, 8]]
    enemy_groups = [[1, 2, 3, 4, 5, 6, 7, 8]]
    # List of 8 enemies
    NUMBER_OF_RUNS = 10
    remaining_runs = get_remaining_runs(f'testdata/{EA_NAME}/', NUMBER_OF_RUNS)
    print(len(remaining_runs))

    # Calculate the number of combinations in each of the 6 parts
    part_size = len(remaining_runs) // 6

    # # Divide the entire population into 6 parts
    # enemy_groups = remaining_runs[:part_size]
    # enemy_groups = remaining_runs[part_size:2 * part_size]
    # enemy_groups = remaining_runs[2 * part_size:3 * part_size]
    # # enemy_groups = remaining_runs[3 * part_size:4 * part_size]
    # # enemy_groups = remaining_runs[4 * part_size:5 * part_size]
    # # enemy_groups = remaining_runs[5 * part_size:]


    # for enemies in enemy_groups:
    #     for run in range(1):  # the amount of runs are represented by repeating an enemy combination
    #         print(f"Running EA with enemies {enemies}, run {run + 1}")
    #         # Initialize the EA object
    #         ea = EA(population_size=population_size,
    #                 n_vars=n_vars,
    #                 upper_bound=upper_bound,
    #                 lower_bound=lower_bound,
    #                 crossover_rate=crossover_rate,
    #                 mutation_rate=mutation_rate,
    #                 mutation_std=mutation_std,
    #                 tournament_size=tournament_size,
    #                 alpha=alpha,
    #                 env=env,
    #                 no_generations=no_generations,
    #                 enemies=enemies)

    #         # Run the evolutionary algorithm
    #         ea.run()


if __name__ == '__main__':
    main()
