import os
import glob
import numpy as np
import matplotlib.pyplot as plt
# from PyQt5 import QtCore, QtGui, QtWidgets

# Function to load mean, max fitness, and variety data from CSV files
def load_fitness_data(ea_folder, enemy_folder):
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
def plot_fitness_statistics(generations, ea1_data, ea2_data, ea1_label, ea2_label, enemy):
    plt.figure(figsize=(12, 12))

    # Extract data for EA1
    avg_max_fitness_ea1, std_max_fitness_ea1, avg_mean_fitness_ea1, std_mean_fitness_ea1, avg_variety_ea1, std_variety_ea1 = ea1_data
    # Extract data for EA2
    avg_max_fitness_ea2, std_max_fitness_ea2, avg_mean_fitness_ea2, std_mean_fitness_ea2, avg_variety_ea2, std_variety_ea2 = ea2_data

    # Plot mean and max fitness on the first subplot EA1
    plt.subplot(2, 1, 1)
    plt.plot(generations, avg_max_fitness_ea1, label=f'Avg Max Fitness {ea1_label}', color='red', linewidth=2)
    plt.plot(generations, avg_mean_fitness_ea1, label=f'Avg Mean Fitness {ea1_label}', color='blue', linewidth=2)

    # Plot the standard deviation for EA1
    plt.fill_between(generations,
                     np.array(avg_max_fitness_ea1) - np.array(std_max_fitness_ea1),
                     np.array(avg_max_fitness_ea1) + np.array(std_max_fitness_ea1),
                     color='red', alpha=0.2, label=f'{ea1_label} Max Fitness Std Dev')

    plt.fill_between(generations,
                     np.array(avg_mean_fitness_ea1) - np.array(std_mean_fitness_ea1),
                     np.array(avg_mean_fitness_ea1) + np.array(std_mean_fitness_ea1),
                     color='blue', alpha=0.2, label=f'{ea1_label} Mean Fitness Std Dev')

    # Plot mean and max fitness for EA2
    plt.plot(generations, avg_max_fitness_ea2, label=f'Avg Max Fitness {ea2_label}', color='orange', linewidth=2, linestyle='--')
    plt.plot(generations, avg_mean_fitness_ea2, label=f'Avg Mean Fitness {ea2_label}', color='purple', linewidth=2, linestyle='--')

    # Plot standard deviation for EA2
    plt.fill_between(generations,
                     np.array(avg_max_fitness_ea2) - np.array(std_max_fitness_ea2),
                     np.array(avg_max_fitness_ea2) + np.array(std_max_fitness_ea2),
                     color='orange', alpha=0.2, label=f'{ea2_label} Max Fitness Std Dev')

    plt.fill_between(generations,
                     np.array(avg_mean_fitness_ea2) - np.array(std_mean_fitness_ea2),
                     np.array(avg_mean_fitness_ea2) + np.array(std_mean_fitness_ea2),
                     color='purple', alpha=0.2, label=f'{ea2_label} Mean Fitness Std Dev')

    plt.title(f'Fitness Statistics for Enemy: {enemy}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()

    # Plot variety 
    plt.subplot(2, 1, 2)
    plt.plot(generations, avg_variety_ea1, label=f'Avg Variety {ea1_label}', color='green', linewidth=2)
    plt.fill_between(generations,
                     np.array(avg_variety_ea1) - np.array(std_variety_ea1),
                     np.array(avg_variety_ea1) + np.array(std_variety_ea1),
                     color='green', alpha=0.2, label=f'{ea1_label} Variety Std Dev')

    plt.plot(generations, avg_variety_ea2, label=f'Avg Variety {ea2_label}', color='cyan', linewidth=2, linestyle='--')
    plt.fill_between(generations,
                     np.array(avg_variety_ea2) - np.array(std_variety_ea2),
                     np.array(avg_variety_ea2) + np.array(std_variety_ea2),
                     color='cyan', alpha=0.2, label=f'{ea2_label} Variety Std Dev')

    plt.title(f'Variety Statistics for Enemy: {enemy}')
    plt.xlabel('Generation')
    plt.ylabel('Variety')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    output_folder = f"plots/EA1_vs_EA2/enemy_{enemy}"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/fitness_and_variety_statistics_enemy{enemy}.png")
    plt.show()
    plt.close()


# Main function to run the plotting
def main():
    base_data_folder = 'testdata'
    ea1 = 'EA1'
    ea2 = 'EA2'
    enemy = '2'  # Change to the specific enemy you want to analyze

    ea1_folder = os.path.join(base_data_folder, ea1)
    ea2_folder = os.path.join(base_data_folder, ea2)

    enemy_folder = enemy

    generations, avg_max_fitness_ea1, avg_mean_fitness_ea1, std_max_fitness_ea1, std_mean_fitness_ea1, avg_variety_ea1, std_variety_ea1 = load_fitness_data(ea1_folder, enemy_folder)
    _, avg_max_fitness_ea2, avg_mean_fitness_ea2, std_max_fitness_ea2, std_mean_fitness_ea2, avg_variety_ea2, std_variety_ea2 = load_fitness_data(ea2_folder, enemy_folder)

    ea1_data = (avg_max_fitness_ea1, std_max_fitness_ea1, avg_mean_fitness_ea1, std_mean_fitness_ea1, avg_variety_ea1, std_variety_ea1)
    ea2_data = (avg_max_fitness_ea2, std_max_fitness_ea2, avg_mean_fitness_ea2, std_mean_fitness_ea2, avg_variety_ea2, std_variety_ea2)

    plot_fitness_statistics(generations, ea1_data, ea2_data, 'EA1', 'EA2', enemy)
    print(f"Plot saved for EA1 vs EA2, Enemy: {enemy}")


if __name__ == '__main__':
    main()
