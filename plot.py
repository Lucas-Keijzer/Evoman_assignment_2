"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This script is used to visualize the fitness and diversity
statistics for the two different EAs. The data is loaded from the CSV files
generated during the experiments. The diversity statistics are then visualized
in a similar way, showing the average and standard deviation of the diversity
values for each enemy. The plots are saved in the plots/EA1_vs_EA2 folder.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

    # Calculate the average and standard deviation for each generation
    avg_max_fitness = [np.mean(fitness) for fitness in max_fitnesses if fitness]  # Filter out empty lists
    avg_mean_fitness = [np.mean(fitness) for fitness in mean_fitnesses if fitness]  # Filter out empty lists
    avg_variety = [np.mean(variety) for variety in varieties if variety]  # Filter out empty lists
    std_max_fitness = [np.std(fitness) for fitness in max_fitnesses if fitness]  # Filter out empty lists
    std_mean_fitness = [np.std(fitness) for fitness in mean_fitnesses if fitness]  # Filter out empty lists
    std_variety = [np.std(variety) for variety in varieties if variety]  # Filter out empty lists

    # Generate corresponding generations list (without gaps)
    valid_generations = [i for i, fitness in enumerate(max_fitnesses) if fitness]

    return valid_generations, avg_max_fitness, avg_mean_fitness, std_max_fitness, std_mean_fitness, avg_variety, std_variety

# Function to plot the fitness statistics for multiple EAs
def plot_fitness(ea1_folder, ea2_folder):
    enemies = ['2', '5', '8']  # List of enemies
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # Create a figure with 1 row and 3 columns

    for i, enemy in enumerate(enemies):
        generations, avg_max_fitness_ea1, avg_mean_fitness_ea1, std_max_fitness_ea1, std_mean_fitness_ea1, _, _ = load_data(ea1_folder, enemy)
        _, avg_max_fitness_ea2, avg_mean_fitness_ea2, std_max_fitness_ea2, std_mean_fitness_ea2, _, _ = load_data(ea2_folder, enemy)

        # Plot EA1
        axs[i].plot(generations, avg_max_fitness_ea1, label=f'Avg Max Fitness {ea1} Enemy {enemy}', color='red', linewidth=2)
        axs[i].plot(generations, avg_mean_fitness_ea1, label=f'Avg Mean Fitness {ea1} Enemy {enemy}', color='blue', linestyle='-', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_max_fitness_ea1) - np.array(std_max_fitness_ea1),
                            np.array(avg_max_fitness_ea1) + np.array(std_max_fitness_ea1),
                            color='red', alpha=0.2)

        axs[i].fill_between(generations,
                            np.array(avg_mean_fitness_ea1) - np.array(std_mean_fitness_ea1),
                            np.array(avg_mean_fitness_ea1) + np.array(std_mean_fitness_ea1),
                            color='blue', alpha=0.2)

        # Plot EA2
        axs[i].plot(generations, avg_max_fitness_ea2, label=f'Avg Max Fitness {ea2} Enemy {enemy}', color='orange', linewidth=2)
        axs[i].plot(generations, avg_mean_fitness_ea2, label=f'Avg Mean Fitness {ea2} Enemy {enemy}', color='purple', linestyle='-', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_max_fitness_ea2) - np.array(std_max_fitness_ea2),
                            np.array(avg_max_fitness_ea2) + np.array(std_max_fitness_ea2),
                            color='orange', alpha=0.2)

        axs[i].fill_between(generations,
                            np.array(avg_mean_fitness_ea2) - np.array(std_mean_fitness_ea2),
                            np.array(avg_mean_fitness_ea2) + np.array(std_mean_fitness_ea2),
                            color='purple', alpha=0.2)

        # Set titles for each subplot
        axs[i].set_title(f'Fitness Statistics for enemy {enemy}', fontsize=16)
        axs[i].grid(True)

        # Create an inset for the zoomed-in view
        axins = inset_axes(axs[i], width="30%", height="30%", loc='lower right', borderpad=2)  # borderpad adjusts the inset's padding
        axins.plot(generations, avg_max_fitness_ea1, color='red', linewidth=2)
        axins.plot(generations, avg_mean_fitness_ea1, color='blue', linestyle='--', linewidth=2)
        axins.plot(generations, avg_max_fitness_ea2, color='orange', linewidth=2)
        axins.plot(generations, avg_mean_fitness_ea2, color='purple', linestyle='--', linewidth=2)

        # Set the limits for the zoomed-in view
        axins.set_ylim(88, 95)
        axins.set_xlim(min(generations), max(generations))

        # Remove x and y ticks and labels on the inset
        axins.set_xticks([])
        axins.set_yticks([])

        # Add grid to the inset
        axins.grid(True)

    # Set a single ylabel for all subplots
    axs[0].set_ylabel('Fitness', fontsize=14)

    # Centralize the x-label for the entire figure
    fig.text(0.5, 0.01, 'Generation', ha='center', fontsize=14)

    # Set a common legend in the bottom right of the figure, slightly higher
    handles, labels = axs[0].get_legend_handles_labels()  # Get legend handles from the first subplot
    fig.legend(handles, labels, loc='lower right', fontsize=10, bbox_to_anchor=(0.33, 0.45))

    # Adjust the layout to increase padding between plots
    plt.tight_layout(pad=4.0)  # Increase padding between subplots

    output_folder = "plots/EA1_vs_EA2/all_enemies"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/fitness_statistics_all_enemies.png")
    plt.show()
    plt.close()


# Function to plot the diversity statistics for multiple EAs
def plot_diversity(ea1_folder, ea2_folder):
    enemies = ['2', '5', '8']  # List of enemies
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # Create a figure with 1 row and 3 columns

    for i, enemy in enumerate(enemies):
        generations, _, _, _, _, avg_diversity_ea1, std_diversity_ea1 = load_data(ea1_folder, enemy)
        _, _, _, _, _, avg_diversity_ea2, std_diversity_ea2 = load_data(ea2_folder, enemy)

        # Plot EA1
        axs[i].plot(generations, avg_diversity_ea1, label=f'Avg Diversity {ea1}', color='green', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_diversity_ea1) - np.array(std_diversity_ea1),
                            np.array(avg_diversity_ea1) + np.array(std_diversity_ea1),
                            color='green', alpha=0.2)

        # Plot EA2
        axs[i].plot(generations, avg_diversity_ea2, label=f'Avg Diversity {ea2}', color='cyan', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_diversity_ea2) - np.array(std_diversity_ea2),
                            np.array(avg_diversity_ea2) + np.array(std_diversity_ea2),
                            color='cyan', alpha=0.2)

        # Set titles for each subplot
        axs[i].set_title(f'Diversity Statistics for Enemy: {enemy}', fontsize=16)
        axs[i].grid(True)

    # Set a single ylabel for all subplots
    axs[0].set_ylabel('Diversity (average euclidean distance)', fontsize=14)

    # Centralize the x-label for the entire figure
    fig.text(0.5, 0.01, 'Generation', ha='center', fontsize=14)

    # Set a common legend in the bottom right of the figure
    handles, labels = axs[0].get_legend_handles_labels()  # Get legend handles from the first subplot
    fig.legend(handles, labels, loc='lower right', fontsize=10, bbox_to_anchor=(0.98, 0.55))

    # Adjust the layout to increase padding between plots
    plt.tight_layout(pad=4.0)  # Increase padding between subplots

    output_folder = "plots/EA1_vs_EA2/all_enemies"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/diversity_statistics_all_enemies.png")
    plt.show()
    plt.close()


def main():
    base_data_folder = 'final_data'

    ea1_folder = os.path.join(base_data_folder, ea1)
    ea2_folder = os.path.join(base_data_folder, ea2)

    plot_fitness(ea1_folder, ea2_folder)
    plot_diversity(ea1_folder, ea2_folder)


if __name__ == '__main__':
    main()
