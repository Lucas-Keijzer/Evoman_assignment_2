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

ea_names = ['EA1', 'EA2']  # run all for both EA's
# ea_names = ['EA1']  # use EA1 for testing

# loop over the ea's and the groups of enemies
for ea in ea_names:
    n_files = 0
    for i, enemies in enumerate(enemy_groups):
        # print(f"Processing enemy group number {i+1}/{len(enemy_groups)} {enemies} for {ea}")
        solutions = []

        enemies_name = ''.join(str(e) for e in enemies)
        directory = f"{folder_name}/{ea}/{enemies_name}/"
        files = os.listdir(directory)
        n_files += len(files)
    print(f"final amount of files: {n_files}")


