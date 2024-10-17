"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This file loads all the best individuals from all the 'n'
experiments done by either EA, and finds the 10 best individuals based on
the 'individual gain' values. It also finds the two best individuals based on
the amount of enemies beaten and one on the individual gain for the competition.
These best few individuals are then saved in a separate folder for further
analysis and comparison.
"""

# standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import itertools

# evoman framework
from evoman.environment import Environment
from demo_controller import player_controller

# imports file utils
from file_utils import save_final_best_solution

# obligatory experiment name
experiment_name = 'test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# Define the number of neurons
n_hidden_neurons = 10

# Define the folder name where the best solutions are stored
folder_name = 'best_solutions'

# Define the number of individuals to save
n_individuals_to_save = 10

# all possible triples of enemies
enemy_groups = list(itertools.combinations(range(1, 9), 3))
# print(len(enemy_groups))
# enemy_groups = [[1, 2, 3]]  # use only the first group for testing comment dit ff uit wanneer je alles runt

enemy_groups = list(itertools.combinations(range(1, 9), 3))
i_s = enemy_groups.index((3,4,8))
i_e = enemy_groups.index((3,5,7))
print(i_s, i_e)


