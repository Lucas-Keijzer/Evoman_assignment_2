# ################################
# # EvoMan FrameWork - V1.0 2016 #
# # Author: Karine Miras         #
# # karine.smiras@gmail.com      #
# ################################

# # imports framework
# import sys, os

# from evoman.environment import Environment

# experiment_name = 'dummy_demo'
# if not os.path.exists(experiment_name):
#     os.makedirs(experiment_name)

# # initializes environment with ai player using random controller, playing against static enemy
# env = Environment(experiment_name=experiment_name)
# env.play()

# imports framework and additional libraries

import sys
import os
import numpy as np
from evoman.environment import Environment
from evoman.controller import Controller

# Create a directory for the experiment
experiment_name = 'neural_network_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Define the Neural Network based Controller
class NeuralNetworkController(Controller):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Initialize random weights for a simple 1-layer network (input -> hidden -> output)
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.num_inputs, self.num_hidden))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.num_hidden, self.num_outputs))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def control(self, inputs, controller=None):
        # Perform forward propagation
        hidden_activation = self.sigmoid(np.dot(inputs, self.weights_input_hidden))
        output_activation = self.sigmoid(np.dot(hidden_activation, self.weights_hidden_output))
        
        # Return actions based on outputs. These could be binary or continuous, but we will assume 0 or 1 actions.
        return [1 if output > 0.5 else 0 for output in output_activation]

# Initialize environment with the neural network controller
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  enemymode="static",
                  player_controller=None,  # We'll pass the player controller during play
                  speed="normal",
                  visuals=True)

# Get the number of sensors (inputs) from the environment
num_sensors = env.get_num_sensors()

# Define the structure of the neural network (e.g., 10 hidden neurons)
num_hidden_neurons = 10
num_actions = 5  # Assuming 5 possible actions (e.g., move left, right, jump, shoot, etc.)

# Create an instance of the Neural Network Controller
neural_network_controller = NeuralNetworkController(num_inputs=num_sensors, num_hidden=num_hidden_neurons, num_outputs=num_actions)

# Play the game using the neural network as the controller
fitness, player_life, enemy_life, time = env.play(pcont=neural_network_controller)

# Print the results
print(f"Fitness: {fitness}, Player life: {player_life}, Enemy life: {enemy_life}, Time: {time}")
