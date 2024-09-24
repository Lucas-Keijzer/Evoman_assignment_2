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

        time_score = -np.log(self.get_time)

        # ensures slower solutions are worse when the player wins
        if self.get_playerlife() <= 0:
            time_score = abs(time_score)

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
mutation_rate = 0.1         # rate of mutation
mutation_std = 0.1          # used in gaussian additive mutation

last_best = 0


# Apply clamping to each gene in the individual
def clamp(individual):
    return np.clip(individual, lower_bound, upper_bound)


# creates random pop with right structure sizes
def initialize_population():
    return np.random.uniform(lower_bound, upper_bound, (population_size, n_vars))


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
        if np.random.uniform(0, 1) < mutation_rate:
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

    # go for set amount of generations
    for generation in range(no_generations):
        fitness_population = evaluate_population(env, population)
        next_generation = []

        # make new offspring until we have a new full generation
        while len(next_generation) < population_size:

            # Select two parents using tournament selection
            p1 = tournament_selection(population, fitness_population)
            p2 = tournament_selection(population, fitness_population)

            # Perform crossover to produce offspring (iucludes mutation)
            offspring = crossover_single(p1, p2)

            # mutation
            offspring = mutate_individual(offspring)

            # Add offspring to the next generation
            next_generation.append(offspring)

        # Replace the old population with the new one
        population = np.array(next_generation)[:population_size]

        # Optional: Log the best fitness for monitoring
        best_fitness = np.max(fitness_population)
        print(f"Generation {generation + 1}/{no_generations}, Best Fitness: {best_fitness}")

    # Final evaluation and results
    final_fitness = evaluate_population(env, population)  # Evaluate final population
    best_solution = population[np.argmax(final_fitness)]
    print("Best Solution:", best_solution)
    print("Best Fitness:", np.max(final_fitness))

    # Save the weights to a file
    np.save('pjotr_weights.npy', best_solution)


if __name__ == '__main__':
    main()