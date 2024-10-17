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


def main():
    # Define the number of neurons
    n_hidden_neurons = 10

    # Define the folder name where the best solutions are stored
    folder_name = 'best_solutions'

    # Define the number of individuals to save
    n_individuals_to_save = 10

    # all possible triples of enemies

    enemy_groups = list(itertools.combinations(range(1, 9), 3))[::-1][:20]
    # print(len(enemy_groups))
    # enemy_groups = [[1, 2, 3]]  # use only the first group for testing comment dit ff uit wanneer je alles runt

    ea_names = ['EA1', 'EA2']  # run all for both EA's
    # ea_names = ['EA1']  # use EA1 for testing

    # loop over the ea's and the groups of enemies
    for ea in ea_names:
        for i, enemies in enumerate(enemy_groups):
            print(f"Processing enemy group number {i+1}/{len(enemy_groups)} {enemies} for {ea}")
            solutions = []

            enemies_name = ''.join(str(e) for e in enemies)
            directory = f"{folder_name}/{ea}/{enemies_name}/"
            files = os.listdir(directory)

            # loop over all best experiment run solutions
            for file in files[:]:
                gains = []
                lives = []
                weights = np.loadtxt(directory + file)

                for enemy in range(1, 9):

                    # create the environment
                    env = Environment(experiment_name=experiment_name,
                                    playermode="ai",
                                    enemies=[enemy],
                                    player_controller=player_controller(n_hidden_neurons),
                                    speed="fastest",
                                    enemymode="static",
                                    level=2,
                                    visuals=False)

                    fitness, player_life, enemy_life, _ = env.play(weights)

                    # Calculate gain for this enemy
                    gain = player_life - enemy_life
                    gains.append(gain)
                    lives.append((player_life, enemy_life))

                total_gain = sum(gains)
                enemies_beaten = [-1 if el > 0 else pl for pl, el in lives]
                number_of_enemies_beaten = len([l for l in enemies_beaten if l > -1])

                # add the individual if it is better than the current worst individual
                if len(solutions) < n_individuals_to_save:
                    solutions.append((total_gain, number_of_enemies_beaten, enemies_beaten, weights))
                else:
                    if total_gain > solutions[0][0]:
                        solutions[0] = (total_gain, number_of_enemies_beaten, enemies_beaten, weights)

                # resort after adding a new individual
                solutions.sort(key=lambda x: x[0])


            # save all the n best solutions based on the individual gain per ea and enemy group
            for i, solution in enumerate(solutions):
                # print(f"Saving solution {i+1} with gain {solution[0]}, beaten enemies {solution[1]}, enemies beaten {solution[2]}")

                save_final_best_solution(weights=solution[-1], individual_gain=solution[0],
                                         number_of_enemies_beaten=solution[1],
                                         enemies_beaten=solution[2], enemies=enemies, ea_name=ea,
                                         name=f"solution{i+1}")

            # get and save the best solution based on the individual gain
            best_solution = solutions[-1]
            save_final_best_solution(weights=best_solution[-1],
                                     individual_gain=best_solution[0],
                                     number_of_enemies_beaten=best_solution[1],
                                     enemies_beaten=best_solution[2], enemies=enemies, ea_name=ea,
                                     name=f"best_gain")

            # get and save the best solution based on the amount of enemies beaten
            solutions.sort(key=lambda x: (x[1], x[0]))
            best_solution = solutions[-1]
            save_final_best_solution(weights=best_solution[-1],
                                     individual_gain=best_solution[0],
                                     number_of_enemies_beaten=best_solution[1],
                                     enemies_beaten=best_solution[2], enemies=enemies, ea_name=ea,
                                     name=f"best_defeated")


if __name__ == '__main__':
    main()
