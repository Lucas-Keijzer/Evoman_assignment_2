"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This file implements the tuning mechanism for the evolutionary
algorithm. Specifically, it tunes the hyperparameters of the EA1 (fitness-based)
algorithm using Optuna. The objective function is the fitness of the best
individual found by the EA1 algorithm. The hyperparameters to be tuned are the
population size, crossover rate, mutation rate, mutation standard deviation,
blend crossover alpha, tournament size, and number of generations. The search
space for each hyperparameter is defined in the `objective` function. The
`run_optuna_study` function executes the Optuna study with a specified number of
trials. The best hyperparameter configuration and fitness are printed at the end
of the study.
"""

# imports framework
import optuna
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
    part_size = int(np.ceil(len(remaining_runs) / 6))

    # Divide all the possible combinations into 6 parts
    # enemy_groups = remaining_runs[:part_size]
    # enemy_groups = remaining_runs[part_size:2 * part_size]
    # enemy_groups = remaining_runs[2 * part_size:3 * part_size]
    # enemy_groups = remaining_runs[3 * part_size:4 * part_size]
    # enemy_groups = remaining_runs[4 * part_size:5 * part_size]
    # enemy_groups = remaining_runs[5 * part_size:]


    for enemies in enemy_groups:
        for run in range(1):  # the amount of runs are represented by repeating an enemy combination
            print(f"Running EA with enemies {enemies}, run {run + 1}")
            # Initialize the EA object
            ea = EA(population_size=population_size,
                    n_vars=n_vars,
                    upper_bound=upper_bound,
                    lower_bound=lower_bound,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    mutation_std=mutation_std,
                    tournament_size=tournament_size,
                    alpha=alpha,
                    env=env,
                    no_generations=no_generations,
                    enemies=enemies)

            # Run the evolutionary algorithm
            ea.run()

class HyperparameterTuner:
    def __init__(self, env, enemy, num_searches):
        self.env = env
        self.enemy = enemy
        self.num_searches = num_searches

    # Define the search space for the hyperparameters
    def random_search_space(self):
        search_space = {
            "population_size": [50, 100, 200],
            "crossover_rate": [0.6, 0.7, 0.8],
            "mutation_rate": [0.05, 0.1, 0.2],
            "mutation_std": [0.1, 0.3, 0.5],
            "alpha": [0.1, 0.5, 0.9],
            "tournament_size": [3, 5, 7],
            "no_generations": [10, 20, 50]
        }

        # Randomly sample from the search space for `num_searches` trials
        random_configs = []
        for _ in range(self.num_searches):
            config = {param: np.random.choice(values) for param, values in search_space.items()}
            random_configs.append(config)

        return random_configs

    def tune(self):
        # Generate random configurations
        configurations = self.random_search_space()

        best_fitness = float('-inf')
        best_config = None

        # Run EA for each configuration
        for config in configurations:
            print(f"Testing configuration: {config}")

            # Create the EA instance with the current hyperparameter configuration
            ea = EA(
                population_size=config["population_size"],
                n_vars=(self.env.get_num_sensors() + 1) * 10 + (10 + 1) * 5,  # Assuming 10 hidden neurons
                upper_bound=1,
                lower_bound=-1,
                crossover_rate=config["crossover_rate"],
                mutation_rate=config["mutation_rate"],
                mutation_std=config["mutation_std"],
                tournament_size=config["tournament_size"],
                alpha=config["alpha"],
                env=self.env,
                no_generations=config["no_generations"],
                enemy=self.enemy
            )

            # Run the EA
            ea.run()

            # Check if this configuration produced a better result
            if ea.best_solution_fitness > best_fitness:
                best_fitness = ea.best_solution_fitness
                best_config = config
                print(f"New best config: {best_config} with fitness: {best_fitness}")

        print("Best hyperparameter configuration found:")
        print(best_config)
        print(f"Best fitness achieved: {best_fitness}")


def objective(trial):
    # Define the search space for hyperparameters
    population_size = trial.suggest_int('population_size', 50, 200, step=2)  # Range of population size
    crossover_rate = trial.suggest_float('crossover_rate', 0.3, 0.95)  # Range of crossover rate
    mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.5)  # Range of mutation rate
    mutation_std = trial.suggest_float('mutation_std', 0.1, 0.6)  # Range of mutation standard deviation
    alpha = trial.suggest_float('alpha', 0.1, 0.9)  # Range of blend crossover alpha
    tournament_size = trial.suggest_int('tournament_size', 3, 12)  # Tournament size
    no_generations = 10  # Number of generations

    # Set up the environment
    n_hidden_neurons = 10
    env = CustomEnvironment(
        experiment_name='optimization_test',
                        enemies=[1],
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=player_controller(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)


    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # Initialize the EA object with the sampled hyperparameters
    ea = EA(
        population_size=population_size,
        n_vars=n_vars,
        upper_bound=1,
        lower_bound=-1,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_std=mutation_std,
        tournament_size=tournament_size,
        alpha=alpha,
        env=env,
        no_generations=no_generations,
        # enemies=[3,6,7]
        enemies=[2,5,7]
    )
# [3,6,7] , [2,5,7]

    # Run the evolutionary algorithm and get the best fitness
    ea.run()

    # Optuna maximizes by default, so we return the negative of fitness if we're minimizing
    # return ea.best_solution_fitness  # Return the best fitness as the objective value
    fitness_population = ea.evaluate_population()
    max_fitness = np.max(fitness_population)

    return max_fitness


# Function to execute the study
def run_optuna_study(n_trials=2):
    study_name = "optuna_fitness_max_257"
    storage_name = f"sqlite:///{study_name}.db"  # Use SQLite storage

    # Create or load an existing study from the SQLite database
    study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage_name, load_if_exists=True)

    study.optimize(objective, n_trials=n_trials)

    print("Study results saved to SQLite database.")

if __name__ == "__main__":
    run_optuna_study(n_trials=250)  # Set the number of trials (iterations)

# if __name__ == '__main__':
#     main()
