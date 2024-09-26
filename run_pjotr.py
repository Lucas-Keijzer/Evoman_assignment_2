###############################################################################
# EvoMan FrameWork - Run Trained Neural Network                               #
# Author: Your Name                                                           #
###############################################################################

# imports framework
import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'pjotr'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10
weights_file = 'pjotr_weights.npy'  # File containing the trained weights

def main():
    # Load the weights from the file
    best_weights = np.load(weights_file)

    # Initializes environment for single objective mode (specialist) with static enemy and AI player
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      speed="normal",
                      enemymode="static",
                      level=2,
                      visuals=True)

    total_gain = 0  # Initialize total gain

    # Tests saved demo solutions for each enemy
    for en in range(1, 9):
        # Update the enemy
        env.update_parameter('enemies', [en])

        # Load specialist controller
        print(f'\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY {en} \n')

        # Play and get the results
        fitness, player_life, enemy_life, _ = env.play(best_weights)

        # Calculate gain for this enemy
        gain = player_life - enemy_life
        total_gain += gain  # Accumulate total gain

        print(f"Enemy {en}: Player Life: {player_life}, Enemy Life: {enemy_life}, Gain: {gain}")

    print(f"\nTotal Gain Across All Enemies: {total_gain}")

if __name__ == '__main__':
    main()
