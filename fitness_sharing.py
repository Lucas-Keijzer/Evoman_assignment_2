###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import time
import csv
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# standard offensive strategy first
gamma = 0.9
beta = 0.1

# Custom environment class to change the fitness function
class CustomEnvironment(Environment):
    def fitness_single(self):
        time_score = 0

        if self.get_playerlife() <= 0:
            time_score = np.log(self.get_time())
        else:
            time_score = -np.log(self.get_time())

        return gamma * (100 - self.get_enemylife()) + beta * self.get_playerlife() + time_score


# EA class for running the evolutionary algorithm
class EA:
    def __init__(self, population_size, n_vars, upper_bound, lower_bound, crossover_rate, mutation_rate, mutation_std, tournament_size, alpha, env, no_generations, enemy):
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
        self.enemy = enemy  # Add the enemy parameter

        # Update environment with the specified enemy
        self.env = env
        self.env.update_parameter('enemies', [self.enemy])

        self.population = self.initialize_population()

        # hardcoded for now
        self.sigma_share = 0.1 * np.sqrt(self.n_vars)
        self.fitness_alpha = 5

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
        selected_indices = np.random.randint(0, self.population.shape[0], self.tournament_size)
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

    # sharing function that calculates the portion of fitness that one should
    # get based on the distance between the individuals.
    def sh(self, one, two):
        d = self.distance(one, two)
        if d < self.sigma_share:
            return 1 - (d / self.sigma_share) ** self.fitness_alpha
        else:
            return 0

    # applies the fitness sharing directly into the fitness
    def fitness_sharing(self, fitness_population):
        new_fitness = np.zeros_like(fitness_population)

        # loop over all individuals
        for i in range(len(self.population)):
            divide_by = 0
            for j in range(len(self.population)):
                divide_by += self.sh(self.population[i], self.population[j])

            # Avoid division by zero
            if divide_by == 0:
                divide_by = 1
            # update the fitness accordingly
            new_fitness[i] = fitness_population[i] / divide_by

        return new_fitness

    # Calculate the average variety (pairwise distance) in the population
    def calculate_variety(self):
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

   # Save fitness statistics to a CSV file
    def save_fitness_stats_to_csv(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        ea = "EA1"  # Updated EA number
        enemy = str(self.enemy)  # Updated enemy number

        # Define the path for the CSV file
        csv_path = f"testdata/{ea}/{enemy}"
        os.makedirs(csv_path, exist_ok=True)
        csv_filename = f"{csv_path}/fitness_stats_enemy{self.enemy}_{timestamp}.csv"  # Updated filename with enemy number

        # Write the fitness statistics to a CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Max Fitness", "Mean Fitness", "Std Fitness", "Variety"])  # Header
            writer.writerows(self.fitness_stats)

        print(f"Fitness statistics saved to {csv_filename}")

    # The main evolutionary algorithm loop is now inside the run method
    def run(self):
        best_solution = None
        best_solution_fitness = float('-inf')

        # Go for the set amount of generations
        for generation in range(self.no_generations):
            fitness_population = self.evaluate_population()
            fitness_population = self.fitness_sharing(fitness_population)
            next_generation = []

            # Make new offspring until we have a new full generation
            while len(next_generation) < self.population_size:
                # Select two parents using tournament selection
                p1 = self.tournament_selection(fitness_population)
                p2 = self.tournament_selection(fitness_population)

                # Perform crossover to produce offspring
                offspring = self.crossover_single(p1, p2)

                # Mutate offspring
                if np.random.uniform(0, 1) < self.mutation_rate:
                    offspring = self.mutate_individual(offspring)

                next_generation.append(offspring)

            # Replace old population with the new one
            self.population = np.array(next_generation)[:self.population_size]

            # Calculate fitness statistics
            generation_max_fitness = np.max(fitness_population)
            generation_mean_fitness = np.mean(fitness_population)
            generation_std_fitness = np.std(fitness_population)
            generation_variety = self.calculate_variety()

            # Store fitness statistics and variety for this generation
            self.store_fitness_stats(generation + 1, generation_max_fitness, generation_mean_fitness, generation_std_fitness, generation_variety)

            # Log the best fitness for monitoring
            if generation_max_fitness > best_solution_fitness:
                best_solution = self.population[np.argmax(fitness_population)]
                best_solution_fitness = generation_max_fitness

            print(f"{generation + 1}/{self.no_generations}, max: {generation_max_fitness}, mean: {generation_mean_fitness}, std: {generation_std_fitness}, variety: {generation_variety}")

        # Final evaluation and results
        fitness_population = self.evaluate_population()  # Evaluate final population
        generation_max_fitness = np.max(fitness_population)
        if generation_max_fitness > best_solution_fitness:
            best_solution = self.population[np.argmax(fitness_population)]
            best_solution_fitness = generation_max_fitness

        print(f"Final Generation: max: {best_solution_fitness}, mean: {np.mean(fitness_population)}, std: {np.std(fitness_population)}, variety: {self.calculate_variety()}")
        print("Best Fitness:", best_solution_fitness)

        # Save the best solution weights to a file
        np.save('best_solution_weights.npy', best_solution)

        # Save the fitness statistics to a CSV file
        self.save_fitness_stats_to_csv()


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
                            enemies=[2],  # Set enemy here
                            playermode="ai",
                            player_controller=player_controller(n_hidden_neurons),  # you can insert your own controller here
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)

    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # EA configuration
    population_size = 100
    no_generations = 10
    upper_bound = 1
    lower_bound = -1
    crossover_rate = 0.7
    alpha = 0.5
    mutation_rate = 0.1
    mutation_std = 0.5
    tournament_size = 5

    enemy = 2  # Set the enemy here

    assert enemy in [2, 5, 8], "Invalid enemy number. Choose from 2, 5, or 8."

    # run one iteration for now
    for i in range(1):
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
                enemy=enemy)  # Pass the enemy value here

        # Run the evolutionary algorithm
        ea.run()


if __name__ == '__main__':
    main()
