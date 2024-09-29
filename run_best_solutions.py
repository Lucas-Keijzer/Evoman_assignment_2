# imports framework
import os
import numpy as np
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'boxplots'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# Load the best solution from a file
def load_best_solution(ea, enemy):
    directory = f"final_best_solutions/{ea}/{enemy}/"
    filename = f"{directory}solution_with_fitness.npz"

    if os.path.exists(filename):
        data = np.load(filename)
        best_solution = data['weights']
        best_solution_fitness = data['fitness']
        print(f"Loaded best solution with fitness: {best_solution_fitness} from {filename}")
        return best_solution, best_solution_fitness
    else:
        # No solution saved yet, initialize with an empty solution and very bad fitness
        return None, float('-inf')

# Function to create boxplots
def plot_boxplots(gains_ea1, gains_ea2, enemies):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, enemy in enumerate(enemies):
        # Prepare data for boxplot
        data = [gains_ea1[enemy], gains_ea2[enemy]]

        # Create the boxplot
        axes[i].boxplot(data, labels=['EA1', 'EA2'])
        axes[i].set_title(f'Enemy {enemy}')
        axes[i].set_ylabel('Gain (Player Life - Enemy Life)')

    plt.tight_layout()
    plt.show()

def main():
    enemies = [2, 5, 8]  # List of enemies to test
    gains_ea1 = {enemy: [] for enemy in enemies}  # To store gains for EA1
    gains_ea2 = {enemy: [] for enemy in enemies}  # To store gains for EA2

    for ea in ['EA1', 'EA2']:
        for enemy in enemies:
            # Load the weights from the file
            best_weights, score = load_best_solution(ea, enemy)

            # Initializes environment for single objective mode (specialist) with static enemy and AI player
            env = Environment(experiment_name=experiment_name,
                              playermode="ai",
                              enemies=[enemy],
                              player_controller=player_controller(n_hidden_neurons),
                              speed="fastest",
                              enemymode="static",
                              level=2,
                              visuals=False)

            # Run the solution 5 times for each enemy
            gains = []
            for run in range(5):
                # Play and get the results
                fitness, player_life, enemy_life, _ = env.play(best_weights)

                # Calculate gain for this enemy
                gain = player_life - enemy_life
                gains.append(gain)

                print(f"EA {ea}, Enemy {enemy}, Run {run+1}: Player Life: {player_life}, Enemy Life: {enemy_life}, Gain: {gain}")

            # Store the gains for plotting
            if ea == 'EA1':
                gains_ea1[enemy] = gains
            else:
                gains_ea2[enemy] = gains

    # Create the boxplots
    plot_boxplots(gains_ea1, gains_ea2, enemies)

if __name__ == '__main__':
    main()
