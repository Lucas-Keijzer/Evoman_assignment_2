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

folder_name = 'final_best_solutions'


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

    # enemies_folder = ''.join([str(enemy) for enemy in trained_enemies])
    all_enemies = range(1, 9) # List of enemies to test agianst

    enemy_groups = [[1, 2, 3, 4, 5, 6, 7, 8]]
    enemy_groups = [[3, 6, 7]]

    # goated agents:EA2:
    # 125 4,8,9
    # 126 0,
    # 138 5,6
    # 145 5,
    # 234 1,6
    # 247 2(5!),3(5!),9
    # 248 0,5,6,7,9
    # 256 2,3,4(5!),9
    # 257 4(5!+180),7,8,
    # 378 5,7,9
    # 457 2,8,9
    # 458 -(miss 7 toevoegen voor generalisatie)


    # for ea in ['EA1', 'EA2']:
    for ea in ['EA1']:
        gainss = []
        for enemy in enemy_groups:
            enemies_folder = ''.join([str(enemy) for enemy in enemy])
            gains = []
            lives = []
            for enemy in all_enemies:

                # Load the weights from the file
                folder_path = f'{folder_name}/{ea}/{enemies_folder}'
                file_name = os.listdir(folder_path)[-1]
                file_path = os.path.join(folder_path, file_name)

                weights = np.loadtxt(file_path)
                #

                weights = np.array(weights)

                # Initializes environment for single objective mode (specialist) with static enemy and AI player
                env = Environment(experiment_name=experiment_name,
                                playermode="ai",
                                enemies=[enemy],
                                player_controller=player_controller(n_hidden_neurons),
                                speed="normal",
                                enemymode="static",
                                level=2,
                                visuals=True)

                # Play and get the results
                fitness, player_life, enemy_life, _ = env.play(weights)

                # Calculate gain for this enemy
                gain = player_life - enemy_life
                gains.append(gain)
                lives.append((player_life, enemy_life))

                print(f"EA {ea}, Enemy {enemy}, Player Life: {player_life}, Enemy Life: {enemy_life}, Gain: {gain}")
            gainss.append(gains)

    print(gainss)
    for i, gains in enumerate(gainss):
        print({i + 1})
        print(gainss)
        print(f'EA1 total gain agianst all enemies: = {sum(gains)}')
        print(f'EA1 beat enemies: {[(i + 1, pl) for i, (pl, el) in enumerate(lives) if el <=0]}')


if __name__ == '__main__':
    main()
