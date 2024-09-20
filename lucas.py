# imports framework and additional libraries
import sys
import os
import numpy as np
import random
from evoman.environment import Environment
from evoman.controller import Controller

# Create a directory for the experiment
experiment_name = 'evolution_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Neural Network Controller class (same as before)
class NeuralNetworkController(Controller):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Randomly initialize the weights of the neural network
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.num_inputs, self.num_hidden))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.num_hidden, self.num_outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def control(self, inputs, controller=None):
        # Perform forward propagation through the neural network
        hidden_activation = self.sigmoid(np.dot(inputs, self.weights_input_hidden))
        output_activation = self.sigmoid(np.dot(hidden_activation, self.weights_hidden_output))
        
        # Binary actions (e.g., left, right, jump, shoot)
        return [1 if output > 0.5 else 0 for output in output_activation]

# Create an individual with a neural network's weights as the genome
class Individual:
    def __init__(self, nn_controller):
        self.nn_controller = nn_controller
        self.fitness = None  # Fitness will be evaluated later

    def evaluate(self, env):
        try:
            # Run the environment with this individual's neural network controller
            fitness, player_life, enemy_life, time = env.play(pcont=self.nn_controller)
            self.fitness = fitness if fitness is not None else 0  # Ensure a default fitness of 0
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            self.fitness = 0  # Set fitness to 0 in case of an error
        return self.fitness


# Genetic Algorithm parameters
population_size = 10
generations = 3
mutation_rate = 0.3
num_hidden_neurons = 10
num_actions = 5

# Initialize environment with neural network controller
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  enemymode="static",
                  speed="normal",
                  visuals=True)

# Get the number of sensors (inputs) from the environment
num_sensors = env.get_num_sensors()

# Create a population of individuals (neural networks)
def create_population():
    population = []
    for _ in range(population_size):
        nn_controller = NeuralNetworkController(num_inputs=num_sensors, num_hidden=num_hidden_neurons, num_outputs=num_actions)
        population.append(Individual(nn_controller))
    return population

# Selection: selects the top individuals based on fitness
def select(population):
    # Sort by fitness (higher is better)
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    # Select top 50% of the population
    selected = population[:population_size // 2]
    return selected

# Crossover: produces offspring by combining weights of two parents
def crossover(parent1, parent2):
    child_nn = NeuralNetworkController(num_sensors, num_hidden_neurons, num_actions)
    
    # Crossover between input-hidden weights
    crossover_point = np.random.randint(0, parent1.weights_input_hidden.size)
    child_nn.weights_input_hidden.flat[:crossover_point] = parent1.weights_input_hidden.flat[:crossover_point]
    child_nn.weights_input_hidden.flat[crossover_point:] = parent2.weights_input_hidden.flat[crossover_point:]

    # Crossover between hidden-output weights
    crossover_point = np.random.randint(0, parent1.weights_hidden_output.size)
    child_nn.weights_hidden_output.flat[:crossover_point] = parent1.weights_hidden_output.flat[:crossover_point]
    child_nn.weights_hidden_output.flat[crossover_point:] = parent2.weights_hidden_output.flat[crossover_point:]
    
    return Individual(child_nn)

# Mutation: mutates the weights of an individual
def mutate(individual):
    if random.random() < mutation_rate:
        # Mutate the weights of the input-hidden layer
        mutation_indices = np.random.randint(0, individual.nn_controller.weights_input_hidden.size, size=2)
        individual.nn_controller.weights_input_hidden.flat[mutation_indices] += np.random.normal(0, 0.1, size=2)
        
        # Mutate the weights of the hidden-output layer
        mutation_indices = np.random.randint(0, individual.nn_controller.weights_hidden_output.size, size=2)
        individual.nn_controller.weights_hidden_output.flat[mutation_indices] += np.random.normal(0, 0.1, size=2)

# Evolutionary algorithm
def evolve():
    # Initialize population
    population = create_population()

    for generation in range(generations):
        print(f"Generation {generation}")

        # Evaluate fitness for each individual
        for individual in population:
            individual.evaluate(env)

        # Ensure all individuals have a valid fitness value
        for individual in population:
            if individual.fitness is None:
                individual.fitness = 0  # Assign 0 if fitness is None
        
        # Select the top-performing individuals
        selected_population = select(population)

        # Create the next generation through crossover and mutation
        next_generation = []
        while len(next_generation) < population_size:
            # Randomly select two parents from the selected individuals
            parent1, parent2 = random.sample(selected_population, 2)

            # Perform crossover to produce a child
            child = crossover(parent1.nn_controller, parent2.nn_controller)

            # Mutate the child
            mutate(child)

            next_generation.append(child)

        # The new population for the next generation
        population = next_generation

        # Print the best fitness from this generation
        best_individual = max(population, key=lambda ind: ind.fitness if ind.fitness is not None else 0)
        print(f"Best fitness of generation {generation}: {best_individual.fitness}")


# Start the evolutionary process
evolve()
