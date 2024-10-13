"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This script is used to execute the best found solutions during
the 10 runs of the two different EAs. The best solutions are saved in the
final_best_solutions folder and are loaded from there. The best solutions are
played against the respective three different enemies with RANDOM INITIALISATION
(might change this back) and the results are visualized in boxplots.
Additionally, a statistical test is performed to determine if the results are
significantly different.
"""

# imports framework
import os
import numpy as np
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller
import scipy.stats as stats
import pandas as pd

experiment_name = 'boxplots'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific exampleload_best_solution
n_hidden_neurons = 10

folder_name = 'best_solutions'

# Load the best solution from a file
def load_best_solution(ea, enemies):
    enemies_name = ''.join(str(e) for e in enemies)
    directory = f"{folder_name}/{ea}/{enemies_name}/"
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


def plot_boxplots(gains_ea1, gains_ea2, enemies):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharey=True)  # sharey=True for shared y-axis

    for i, enemy in enumerate(enemies):
        # Prepare data for boxplot
        data = [gains_ea1[enemy], gains_ea2[enemy]]

        # Create the boxplot
        axes[i].boxplot(data, labels=['EA1', 'EA2'])

        # Increase font sizes
        axes[i].set_title(f'Enemy {enemy}', fontsize=28)  # Increase title size
        axes[i].tick_params(axis='y', labelsize=12)  # Increase y-axis tick label size
        axes[i].tick_params(axis='x', labelsize=12)  # Increase x-axis tick label size

    # Set shared y-axis label, placing it outside of the plot
    fig.text(0.03, 0.5, 'Gain (Player Life - Enemy Life)', va='center', rotation='vertical', fontsize=24)

    # Add a title for the entire figure
    fig.suptitle('Boxplots of 5 Runs per EA per Enemy', fontsize=30)

    # Adjust layout to prevent overlap and leave space for the y-label and title
    plt.subplots_adjust(left=0.06, right=0.95, top=0.85, bottom=0.1)  # Increased top to fit the title
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Ensure layout is adjusted for y-label and title

    # Save the plot in the existing 'plots' folder as 'boxplot.png'
    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/boxplot.png")

    plt.show()
    plt.close()


def statistical_test(gains_ea1, gains_ea2, enemies):
    results = []

    for enemy in enemies:
        # Gather data for each enemy
        data_ea1 = gains_ea1[enemy]
        data_ea2 = gains_ea2[enemy]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data_ea1, data_ea2)

        # Calculate means and 95% confidence intervals
        mean_ea1 = np.mean(data_ea1)
        mean_ea2 = np.mean(data_ea2)

        ci_ea1 = stats.t.interval(0.95, len(data_ea1)-1, loc=mean_ea1, scale=stats.sem(data_ea1))
        ci_ea2 = stats.t.interval(0.95, len(data_ea2)-1, loc=mean_ea2, scale=stats.sem(data_ea2))

        results.append({
            'Enemy': enemy,
            'Mean EA1': mean_ea1,
            'Mean EA2': mean_ea2,
            'T-Statistic': t_stat,
            'P-Value': p_value,
            '95% CI EA1': ci_ea1,
            '95% CI EA2': ci_ea2,
            'Significant Difference': p_value < 0.05  # True if significant
        })

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)

    return results_df


def main():
    trained_enemies = [2, 5, 8]
    enemies = [i for i in range(1, 9)]  # all enemies
    gains_ea1 = {enemy: [] for enemy in enemies}  # To store gains for EA1
    gains_ea2 = {enemy: [] for enemy in enemies}  # To store gains for EA2

    enemies = range(1, 9) # List of enemies to test agianst

    no_runs = 1

    for ea in ['EA1', 'EA2']:
    # for ea in ['EA1']:
        for enemy in enemies:
            # Load the weights from the file
            best_weights, score = load_best_solution(ea, trained_enemies)

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
            for run in range(no_runs):
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

    # # Create the boxplots
    # plot_boxplots(gains_ea1, gains_ea2, enemies)

    # # Perform statistical tests and display results
    # statistical_results = statistical_test(gains_ea1, gains_ea2, enemies)

    if gains_ea1:
        print(gains_ea1.values())
        print(f'EA1 total gain agianst enemies: {list(enemies)} = {sum([sum(el) for el in gains_ea1.values()])}')
    if gains_ea2:
        print(gains_ea2.values())
        print(f'EA2 total gain agianst enemies: {list(enemies)} = {sum([sum(el) for el in gains_ea2.values()])}')

    # print(statistical_results)


if __name__ == '__main__':
    main()
