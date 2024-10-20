"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This script is used to visualize the fitness and diversity
statistics for the two different EAs. The data is loaded from the CSV files
generated during the experiments. The diversity statistics are then visualized
in a similar way, showing the average and standard deviation of the diversity
values for each enemy. It also plots the best solutions in a boxplot and
performs a statistical analysis on all 4 different setups to determine if the
differences are significant.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from file_utils import load_best_plot_solutions
import pandas as pd
import scipy.stats as stats
from itertools import product

ea1 = 'EA1'
ea2 = 'EA2'


# Function to load mean, max fitness, and variety data from CSV files
def load_data(ea_folder, enemy_folder):
    csv_files = glob.glob(os.path.join(ea_folder, enemy_folder, '*.csv'))
    generations = []
    max_fitnesses = []
    mean_fitnesses = []
    varieties = []  # To store variety data

    for file in csv_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            # Skip the header
            for line in lines[1:]:
                generation, max_fitness, mean_fitness, _, variety = line.strip().split(',')
                generation = int(generation)
                max_fitness = float(max_fitness)
                mean_fitness = float(mean_fitness)
                variety = float(variety)

                # Ensure we have enough space in our lists
                while len(max_fitnesses) <= generation:
                    max_fitnesses.append([])  # Create a new list for this generation
                while len(mean_fitnesses) <= generation:
                    mean_fitnesses.append([])  # Create a new list for this generation
                while len(varieties) <= generation:
                    varieties.append([])  # Create a new list for this generation

                # Append fitness and variety values to the respective generation
                max_fitnesses[generation].append(max_fitness)
                mean_fitnesses[generation].append(mean_fitness)
                varieties[generation].append(variety)

    # Calculate the average and standard deviation for each generation and filter out empty lists
    avg_max_fitness = [np.mean(fitness) for fitness in max_fitnesses if fitness]
    avg_mean_fitness = [np.mean(fitness) for fitness in mean_fitnesses if fitness]
    avg_variety = [np.mean(variety) for variety in varieties if variety]
    std_max_fitness = [np.std(fitness) for fitness in max_fitnesses if fitness]
    std_mean_fitness = [np.std(fitness) for fitness in mean_fitnesses if fitness]
    std_variety = [np.std(variety) for variety in varieties if variety]

    # Generate corresponding generations list (without gaps)
    valid_generations = [i for i, fitness in enumerate(max_fitnesses) if fitness]

    return valid_generations, avg_max_fitness, avg_mean_fitness, std_max_fitness, std_mean_fitness, avg_variety, std_variety


# Function to plot the fitness statistics for both eas trained on both enemies
def plot_fitness(ea1_folder, ea2_folder):
    enemies = ['257', '367']  # two training groups
    fig, axs = plt.subplots(2, 1, figsize=(16, 24))

    for i, enemy in enumerate(enemies):
        generations, avg_max_fitness_ea1, avg_mean_fitness_ea1, std_max_fitness_ea1, std_mean_fitness_ea1, _, _ = load_data(ea1_folder, enemy)
        _, avg_max_fitness_ea2, avg_mean_fitness_ea2, std_max_fitness_ea2, std_mean_fitness_ea2, _, _ = load_data(ea2_folder, enemy)


        # scale the values by 1/2 (since we changed the fitness function to be 0-200)
        avg_max_fitness_ea1 = [x / 2 for x in avg_max_fitness_ea1]
        avg_mean_fitness_ea1 = [x / 2 for x in avg_mean_fitness_ea1]
        std_max_fitness_ea1 = [x / 2 for x in std_max_fitness_ea1]
        std_mean_fitness_ea1 = [x / 2 for x in std_mean_fitness_ea1]

        avg_max_fitness_ea2 = [x / 2 for x in avg_max_fitness_ea2]
        avg_mean_fitness_ea2 = [x / 2 for x in avg_mean_fitness_ea2]
        std_max_fitness_ea2 = [x / 2 for x in std_max_fitness_ea2]
        std_mean_fitness_ea2 = [x / 2 for x in std_mean_fitness_ea2]

        # show only the relevant amount of generations
        if i == 0:
            points = 12
            generations = generations[:points]
            avg_max_fitness_ea1 = avg_max_fitness_ea1[:points]
            avg_mean_fitness_ea1 = avg_mean_fitness_ea1[:points]
            std_max_fitness_ea1 = std_max_fitness_ea1[:points]
            std_mean_fitness_ea1 = std_mean_fitness_ea1[:points]

            avg_max_fitness_ea2 = avg_max_fitness_ea2[:points]
            avg_mean_fitness_ea2 = avg_mean_fitness_ea2[:points]
            std_max_fitness_ea2 = std_max_fitness_ea2[:points]
            std_mean_fitness_ea2 = std_mean_fitness_ea2[:points]
        elif i == 1:
            points = 20
            generations = generations[:points]
            avg_max_fitness_ea1 = avg_max_fitness_ea1[:points]
            avg_mean_fitness_ea1 = avg_mean_fitness_ea1[:points]
            std_max_fitness_ea1 = std_max_fitness_ea1[:points]
            std_mean_fitness_ea1 = std_mean_fitness_ea1[:points]

            avg_max_fitness_ea2 = avg_max_fitness_ea2[:points]
            avg_mean_fitness_ea2 = avg_mean_fitness_ea2[:points]
            std_max_fitness_ea2 = std_max_fitness_ea2[:points]
            std_mean_fitness_ea2 = std_mean_fitness_ea2[:points]


        # Plot EA1
        axs[i].plot(generations, avg_max_fitness_ea1, label=f'Avg Max Fitness EA1: fitness based',
                    color='orangered', linewidth=2)
        axs[i].plot(generations, avg_mean_fitness_ea1, label=f'Avg Mean Fitness EA1: fitness based'
                    , color='red', linestyle='-', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_max_fitness_ea1) - np.array(std_max_fitness_ea1),
                            np.array(avg_max_fitness_ea1) + np.array(std_max_fitness_ea1),
                            color='orangered', alpha=0.1)

        axs[i].fill_between(generations,
                            np.array(avg_mean_fitness_ea1) - np.array(std_mean_fitness_ea1),
                            np.array(avg_mean_fitness_ea1) + np.array(std_mean_fitness_ea1),
                            color='red', alpha=0.1)

        # Plot EA2
        axs[i].plot(generations, avg_max_fitness_ea2, label=f'Avg Max Fitness EA2: age based',
                    color='cornflowerblue', linewidth=2)
        axs[i].plot(generations, avg_mean_fitness_ea2, label=f'Avg Mean Fitness EA2: age based',
                    color='blue', linestyle='-', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_max_fitness_ea2) - np.array(std_max_fitness_ea2),
                            np.array(avg_max_fitness_ea2) + np.array(std_max_fitness_ea2),
                            color='cornflowerblue', alpha=0.1)

        axs[i].fill_between(generations,
                            np.array(avg_mean_fitness_ea2) - np.array(std_mean_fitness_ea2),
                            np.array(avg_mean_fitness_ea2) + np.array(std_mean_fitness_ea2),
                            color='blue', alpha=0.1)

        group = ', '.join(enemy)
        # Set titles for each subplot
        axs[i].set_title(f'EAs trained on enemies {group}', fontsize=22)  # Decreased title size
        axs[i].grid(True)

        # Set font size for both axes
        axs[i].tick_params(axis='both', which='major', labelsize=15)

    # Positioning the shared ylabel
    fig.text(0.04, 0.45, 'Fitness', fontsize=20, va='center', rotation='vertical')

    # Centralize the x-label for the entire figure
    fig.text(0.5, 0.05, 'Generation', ha='center', fontsize=25)  # Decreased font size for xlabel

    # Set a common legend in a clear spot on the figure
    handles, labels = axs[0].get_legend_handles_labels()  # Get legend handles from the first subplot
    fig.legend(handles, labels, loc='upper center', fontsize=16, bbox_to_anchor=(0.5, 0.94), ncol=2)

    # Adjust the layout to increase padding between plots
    plt.tight_layout(pad=15.0)
    plt.subplots_adjust(top=0.80)

    # adjust left and right padding
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

    fig.suptitle('Mean and max fitness per generation for both EAs trained on both training groups across 10 runs.', fontsize=25)
    plt.show()


# Function to plot the diversity statistics for one enemy group
def plot_diversity(ea1_folder, ea2_folder):
    enemy = '367'
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))  # Create a figure with 1 row and 1 column

    # Load data for Enemy 2
    generations, _, _, _, _, avg_diversity_ea1, std_diversity_ea1 = load_data(ea1_folder, enemy)
    _, _, _, _, _, avg_diversity_ea2, std_diversity_ea2 = load_data(ea2_folder, enemy)

    # Plot EA1
    ax.plot(generations, avg_diversity_ea1, label=f'Avg Diversity EA1', color='green', linewidth=2)
    ax.fill_between(generations,
                    np.array(avg_diversity_ea1) - np.array(std_diversity_ea1),
                    np.array(avg_diversity_ea1) + np.array(std_diversity_ea1),
                    color='green', alpha=0.2)

    # Plot EA2
    ax.plot(generations, avg_diversity_ea2, label=f'Avg Diversity EA2', color='darkturquoise', linewidth=2)
    ax.fill_between(generations,
                    np.array(avg_diversity_ea2) - np.array(std_diversity_ea2),
                    np.array(avg_diversity_ea2) + np.array(std_diversity_ea2),
                    color='darkturquoise', alpha=0.2)

    # add comma between each enemy
    group = ', '.join(enemy)

    # Set title for the plot
    ax.set_title(f'Diversity statistics per generation for both EAs trained on enemies {group} across 10 runs', fontsize=25)

    # Set labels and grid
    ax.set_ylabel('Diversity (average euclidean distance)', fontsize=20, labelpad=20)
    ax.set_xlabel('Generation', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True)

    # Set a common legend in the top right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(0.95, 0.80), fontsize=14)

    # Adjust the layout to increase padding
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    plt.show()

    # Save the plot
    output_folder = "plots/EA1_vs_EA2/enemy_2"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/diversity_statistics_enemy_2.png")
    plt.close()


# plots the 4 boxplots of both EAs on both enemies from the final_best_solutions folder
# it creates the boxplots for the total gain value of both agents for both enemies
def boxplots():
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a single figure for all 4 boxplots

    # Initialize a list to store gains for plotting and their labels
    all_gains = []
    labels = []

    # Loop over both 'EA1' and 'EA2' and both enemies to load gains
    for ea_name in ['EA1', 'EA2']:
        for enemy in ['257', '367']:
            # Load the best solutions for the current EA and enemy
            best_solutions = list(load_best_plot_solutions(ea_name, [int(el) for el in enemy]))

            # Extract the individual gains from the best solutions and store them
            gains = [solution[0] for solution in best_solutions]
            all_gains.append(gains)

            # Add the label for this boxplot
            labels.append(f'{ea_name}, Enemy {enemy}')

    # Plot the boxplots for each combination of EA and enemy
    ax.boxplot(all_gains, labels=labels)
    ax.set_title('Individual gain boxplots of 10 runs for both EAs trained on both enemy groups',
                 fontsize=25)

    # Set ylabel and adjust its position slightly to the left
    ylabel = ax.set_ylabel('Individual gain', fontsize=20, labelpad=0)
    ylabel.set_position((-0.2, 0.5))

    ax.tick_params(axis='both', which='major', labelsize=15)

    # Manually adjust the axes position for tighter control
    ax.set_position([0.2, 0.2, 0.6, 0.6])  # [left, bottom, width, height]

    # Optionally adjust the subplot parameters to minimize space
    plt.tight_layout(pad=3.0)  # You can adjust this padding

    # Display the plot
    plt.show()


# performs a t-test on the individual gains of all 4 EA setups and prints the results
# of these tests.
def statistical_test():
    # List of enemies
    enemies = ['257', '367']

    # Initialize dictionaries to store EA1 and EA2 gains for each enemy
    gains_ea1 = {enemy: [] for enemy in enemies}
    gains_ea2 = {enemy: [] for enemy in enemies}

    # Load the gains for both EA1 and EA2 for each enemy
    for enemy in enemies:
        for ea_name in [ea1, ea2]:
            best_solutions = list(load_best_plot_solutions(ea_name, [int(el) for el in enemy]))
            gains = [solution[0] for solution in best_solutions]
            if ea_name == ea1:
                gains_ea1[enemy] = gains
            else:
                gains_ea2[enemy] = gains

    # Prepare a list to store the results
    results = []

    # Get all combinations of enemies and EA setups for pairwise comparison
    setups = ['ea1', 'ea2']
    for (enemy1, enemy2), (setup1, setup2) in product(product(enemies, repeat=2), product(setups, repeat=2)):
        if enemy1 == enemy2 and setup1 == setup2:
            # Skip comparisons of the same setup with itself
            continue

        # Get the appropriate data for the given enemy and setup combination
        data_setup1 = gains_ea1[enemy1] if setup1 == 'ea1' else gains_ea2[enemy1]
        data_setup2 = gains_ea1[enemy2] if setup2 == 'ea1' else gains_ea2[enemy2]

        # Perform the t-test between the two setups
        t_stat, p_value = stats.ttest_ind(data_setup1, data_setup2)

        # Calculate means and 95% confidence intervals for both setups
        mean_setup1 = np.mean(data_setup1)
        mean_setup2 = np.mean(data_setup2)

        ci_setup1 = stats.t.interval(0.95, len(data_setup1)-1, loc=mean_setup1, scale=stats.sem(data_setup1))
        ci_setup2 = stats.t.interval(0.95, len(data_setup2)-1, loc=mean_setup2, scale=stats.sem(data_setup2))

        # Append the results for this comparison
        results.append({
            'enemy1': enemy1,
            'setup1': setup1,
            'enemy2': enemy2,
            'setup2': setup2,
            't_stat': t_stat,
            'p_value': p_value,
            'mean_setup1': mean_setup1,
            'mean_setup2': mean_setup2,
            'ci_setup1': ci_setup1,
            'ci_setup2': ci_setup2
        })

    # Convert results to a DataFrame and print it
    results_df = pd.DataFrame(results)
    print(results_df)


def main():
    base_data_folder = 'final_data'

    ea1_folder = os.path.join(base_data_folder, ea1)
    ea2_folder = os.path.join(base_data_folder, ea2)

    # plot_fitness(ea1_folder, ea2_folder)
    # plot_diversity(ea1_folder, ea2_folder)
    # boxplots()
    statistical_test()


if __name__ == '__main__':
    main()
