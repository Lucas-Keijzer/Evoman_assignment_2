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
    fig, axs = plt.subplots(3, 1, figsize=(16, 24))  # Create a figure with 3 rows and 1 column

    for i, enemy in enumerate(enemies):
        generations, avg_max_fitness_ea1, avg_mean_fitness_ea1, std_max_fitness_ea1, std_mean_fitness_ea1, _, _ = load_data(ea1_folder, enemy)
        _, avg_max_fitness_ea2, avg_mean_fitness_ea2, std_max_fitness_ea2, std_mean_fitness_ea2, _, _ = load_data(ea2_folder, enemy)

        # Plot EA1
        axs[i].plot(generations, avg_max_fitness_ea1, label=f'Avg Max Fitness EA1', color='red', linewidth=2)
        axs[i].plot(generations, avg_mean_fitness_ea1, label=f'Avg Mean Fitness EA1', color='blue', linestyle='-', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_max_fitness_ea1) - np.array(std_max_fitness_ea1),
                            np.array(avg_max_fitness_ea1) + np.array(std_max_fitness_ea1),
                            color='red', alpha=0.2)

        axs[i].fill_between(generations,
                            np.array(avg_mean_fitness_ea1) - np.array(std_mean_fitness_ea1),
                            np.array(avg_mean_fitness_ea1) + np.array(std_mean_fitness_ea1),
                            color='blue', alpha=0.2)

        # Plot EA2
        axs[i].plot(generations, avg_max_fitness_ea2, label=f'Avg Max Fitness EA2', color='orange', linewidth=2)
        axs[i].plot(generations, avg_mean_fitness_ea2, label=f'Avg Mean Fitness EA2', color='purple', linestyle='-', linewidth=2)
        axs[i].fill_between(generations,
                            np.array(avg_max_fitness_ea2) - np.array(std_max_fitness_ea2),
                            np.array(avg_max_fitness_ea2) + np.array(std_max_fitness_ea2),
                            color='orange', alpha=0.2)

        axs[i].fill_between(generations,
                            np.array(avg_mean_fitness_ea2) - np.array(std_mean_fitness_ea2),
                            np.array(avg_mean_fitness_ea2) + np.array(std_mean_fitness_ea2),
                            color='purple', alpha=0.2)

        # Set titles for each subplot
        axs[i].set_title(f'Fitness statistics for enemy {enemy}', fontsize=20)  # Decreased title size
        axs[i].grid(True)

        # Set font size for both axes
        axs[i].tick_params(axis='both', which='major', labelsize=12)

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
    axs[1].set_ylabel('Fitness', fontsize=20)  # Decreased font size for ylabel

    # Centralize the x-label for the entire figure
    fig.text(0.5, 0.01, 'Generation', ha='center', fontsize=20)  # Decreased font size for xlabel

    # Set a common legend in a clear spot on the figure
    handles, labels = axs[0].get_legend_handles_labels()  # Get legend handles from the first subplot
    fig.legend(handles, labels, loc='upper center', fontsize=14, bbox_to_anchor=(0.5, 0.94), ncol=2)

    # Adjust the layout to increase padding between plots
    plt.tight_layout(pad=10.0)  # Increased padding between subplots for clarity
    plt.subplots_adjust(top=0.80)

    # adjust left and right padding
    plt.subplots_adjust(left=0.1, right=0.95)

    fig.suptitle('Fitness comparison between EA1 and EA2 over 10 runs per enemy', fontsize=16)

    # Save the plot
    plt.show()

    output_folder = "plots/EA1_vs_EA2/all_enemies"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/fitness_statistics_all_enemies.png")
    plt.close()

# Function to plot the diversity statistics for Enemy 2
def plot_diversity(ea1_folder, ea2_folder):
    enemy = '2'  # Only plot for Enemy 2
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
    ax.plot(generations, avg_diversity_ea2, label=f'Avg Diversity EA2', color='cyan', linewidth=2)
    ax.fill_between(generations,
                    np.array(avg_diversity_ea2) - np.array(std_diversity_ea2),
                    np.array(avg_diversity_ea2) + np.array(std_diversity_ea2),
                    color='cyan', alpha=0.2)

    # Set title for the plot
    ax.set_title(f'Diversity Statistics for enemy {enemy}', fontsize=24)

    # Set labels and grid
    ax.set_ylabel('Diversity (average euclidean distance)', fontsize=16)
    ax.set_xlabel('Generation', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
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



def main():
    base_data_folder = 'final_data'

    ea1_folder = os.path.join(base_data_folder, ea1)
    ea2_folder = os.path.join(base_data_folder, ea2)

    plot_fitness(ea1_folder, ea2_folder)
    plot_diversity(ea1_folder, ea2_folder)


if __name__ == '__main__':
    main()
