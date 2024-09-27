import os
import glob
import numpy as np
import matplotlib.pyplot as plt

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

# Function to plot fitness statistics and variety
def plot_fitness_statistics(generations, avg_max_fitness, std_max_fitness, avg_mean_fitness, std_mean_fitness, avg_variety, std_variety, ea, enemy):
    plt.figure(figsize=(12, 12))

    # Plot mean and max fitness on the first subplot
    plt.subplot(2, 1, 1)
    plt.plot(generations, avg_max_fitness, label='Avg Max Fitness', color='red', linewidth=2)
    plt.plot(generations, avg_mean_fitness, label='Avg Mean Fitness', color='blue', linewidth=2)

    # Plot the shading for standard deviation
    plt.fill_between(generations,
                     np.array(avg_max_fitness) - np.array(std_max_fitness),
                     np.array(avg_max_fitness) + np.array(std_max_fitness),
                     color='red', alpha=0.2, label='Max Fitness Std Dev')

    plt.fill_between(generations,
                     np.array(avg_mean_fitness) - np.array(std_mean_fitness),
                     np.array(avg_mean_fitness) + np.array(std_mean_fitness),
                     color='blue', alpha=0.2, label='Mean Fitness Std Dev')

    plt.title(f'Fitness Statistics for EA: {ea}, Enemy: {enemy}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()

    # Plot variety on the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(generations, avg_variety, label='Avg Variety', color='green', linewidth=2)

    # Plot the shading for standard deviation
    plt.fill_between(generations,
                     np.array(avg_variety) - np.array(std_variety),
                     np.array(avg_variety) + np.array(std_variety),
                     color='green', alpha=0.2, label='Variety Std Dev')

    plt.title(f'Variety Statistics for EA: {ea}, Enemy: {enemy}')
    plt.xlabel('Generation')
    plt.ylabel('Variety')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    output_folder = f"plots/{ea}/{enemy}"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/fitness_and_variety_statistics_enemy{enemy}.png")
    plt.show()
    plt.close()

# Main function to run the plotting
def main():
    base_data_folder = 'testdata'
    ea = 'EA1'  # Change to the specific EA you want to analyze
    enemy = '2'  # Change to the specific enemy you want to analyze

    ea_folder = os.path.join(base_data_folder, ea)
    enemy_folder = enemy

    # Load the fitness and variety data
    generations, avg_max_fitness, avg_mean_fitness, std_max_fitness, std_mean_fitness, avg_variety, std_variety = load_fitness_data(ea_folder, enemy_folder)

    # Plot the statistics
    plot_fitness_statistics(generations, avg_max_fitness, std_max_fitness, avg_mean_fitness, std_mean_fitness, avg_variety, std_variety, ea, enemy)
    print(f"Plot saved for EA: {ea}, Enemy: {enemy}")


if __name__ == '__main__':
    main()
