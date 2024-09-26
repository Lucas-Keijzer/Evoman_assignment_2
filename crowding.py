###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# third change

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# standard offensive strategy first
gamma = 0.9
beta = 0.1

# change the fitness function to invert the time when the player lost
class environm(Environment):

    def fitness_single(self):

        time_score = 0

        if self.get_playerlife() <= 0:
            time_score = np.log(self.get_time)
        else:
            time_score = -np.log(self.get_time)

        return gamma * (100 - self.get_enemylife()) + beta * self.get_playerlife() + time_score


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate_population(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# define global variables
n_vars = 0

upper_bound = 1             # upper bound of the weights in the nn
lower_bound = -1            # lower bound of the weights in the nn
population_size = 100       # pop size
no_generations = 10

tournament_size = 5         # tournament size

crossover_rate = 0.7        # rate of crossover
alpha = 0.5                 # distance from p1 and p2 in blend crossover
mutation_rate = 0.3       # rate of mutation
mutation_std = 0.5          # used in gaussian additive mutation


# Apply clamping to each gene in the individual
def clamp(individual):
    return np.clip(individual, lower_bound, upper_bound)


# creates random pop with right structure sizes
def initialize_population():
    return np.random.uniform(lower_bound, upper_bound, (population_size, n_vars))


# diversity calculation
def calculate_diversity(population):
    total_distance = 0
    count = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = np.linalg.norm(population[i] - population[j])
            total_distance += distance
            count += 1

    return total_distance / count if count > 0 else 0

# selects the best out of 'tournament_size' random indivuals and returns this
# individual.
def tournament_selection(population, fitness_population):
    selected_indices = np.random.randint(0, population.shape[0], tournament_size)
    selected_fitness = fitness_population[selected_indices]
    best_index = np.argmax(selected_fitness)
    return population[selected_indices[best_index]]


# creates one new offspring based on blend crossover method where the alpha
# ensures the offspring is not too close to either of the parents.
def crossover_single(parent1, parent2):
    offspring = np.zeros_like(parent1)

    # Perform blend crossover for each gene
    for i in range(len(parent1)):
        if np.random.uniform(0, 1) > crossover_rate:
            # Perform blend crossover
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            offspring[i] = np.random.uniform(min_val - alpha * range_val, max_val + alpha * range_val)
        else:
            # No crossover, copy the gene directly from one parent
            offspring[i] = parent1[i]

    return clamp(offspring)


# mutates one individual based on gaussian additive mutation method
def mutate_individual(individual):
    for i in range(len(individual)):
        individual[i] += np.random.normal(0, mutation_std)

    return clamp(individual)


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    global n_vars
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    population = initialize_population()

    best_solution = []
    best_solution_fitness = 0

    # go for set amount of generations
    for generation in range(no_generations):
        # Evaluate fitness of the current population
        fitness_population = evaluate_population(env, population)

        # Create a new generation with deterministic crowding
        next_generation = np.copy(population)
        
        # Randomly pair parents
        parent_pairs = np.random.permutation(population_size).reshape(-1, 2)

        for pair in parent_pairs:
            p1, p2 = population[pair[0]], population[pair[1]]

            # Create two offspring via crossover
            offspring1 = crossover_single(p1, p2)
            offspring2 = crossover_single(p1, p2)

            # Mutate the offspring
            offspring1 = mutate_individual(offspring1)
            offspring2 = mutate_individual(offspring2)

            # Calculate distances (Euclidean) between parents and offspring
            d_p1_o1 = np.linalg.norm(p1 - offspring1)
            d_p1_o2 = np.linalg.norm(p1 - offspring2)
            d_p2_o1 = np.linalg.norm(p2 - offspring1)
            d_p2_o2 = np.linalg.norm(p2 - offspring2)

            # Calculate fitness for offspring
            fitness_offspring1 = simulation(env, offspring1)
            fitness_offspring2 = simulation(env, offspring2)

            # Pair offspring with the most similar parent
            if d_p1_o1 + d_p2_o2 < d_p1_o2 + d_p2_o1:
                # Pair offspring1 with p1 and offspring2 with p2
                if fitness_offspring1 > fitness_population[pair[0]]:
                    next_generation[pair[0]] = offspring1  # Replace p1 with offspring1 if offspring is better
                if fitness_offspring2 > fitness_population[pair[1]]:
                    next_generation[pair[1]] = offspring2  # Replace p2 with offspring2 if offspring is better
            else:
                # Pair offspring1 with p2 and offspring2 with p1
                if fitness_offspring1 > fitness_population[pair[1]]:
                    next_generation[pair[1]] = offspring1  # Replace p2 with offspring1 if offspring is better
                if fitness_offspring2 > fitness_population[pair[0]]:
                    next_generation[pair[0]] = offspring2  # Replace p1 with offspring2 if offspring is better

        # Replace  generation
        population = next_generation

        # Log the best fitness for monitoring
        generation_best_fitness = np.max(fitness_population)
        if generation_best_fitness > best_solution_fitness:
            best_solution = population[np.argmax(fitness_population)]
            best_solution_fitness = generation_best_fitness
        print(f"{generation + 1}/{no_generations}, max: {generation_best_fitness}, mean: {np.mean(fitness_population)}, std: {np.std(fitness_population)}")
        diversity = calculate_diversity(population)
        print(f"Generation {generation + 1}: Diversity = {diversity}")

    # Final evaluation and results
    fitness_population = evaluate_population(env, population)  # Evaluate final population
    generation_best_fitness = np.max(fitness_population)
    if generation_best_fitness > best_solution_fitness:
        best_solution = population[np.argmax(fitness_population)]
        best_solution_fitness = generation_best_fitness

    # Save the weights to a file
    np.save('crowding_weights.npy', best_solution)



if __name__ == '__main__':
    main()