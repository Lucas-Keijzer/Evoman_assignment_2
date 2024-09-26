import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Function to load mean and max fitness data from CSV files
def load_fitness_data(ea_folder, enemy_folder):
    csv_files = glob.glob(os.path.join(ea_folder, enemy_folder, '*.csv'))
    generations = []
    max_fitnesses = []
    mean_fitnesses = []

    for file in csv_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            # Skip the header
            for line in lines[1:]:
                generation, max_fitness, mean_fitness, _, _ = line.strip().split(',')
                generation = int(generation)
                max_fitness = float(max_fitness)
                mean_fitness = float(mean_fitness)

                # Ensure we have enough space in our lists
                while len(max_fitnesses) <= generation:
                    max_fitnesses.append([])  # Create a new list for this generation
                while len(mean_fitnesses) <= generation:
                    mean_fitnesses.append([])  # Create a new list for this generation

                # Append fitness values to the respective generation
                max_fitnesses[generation].append(max_fitness)
                mean_fitnesses[generation].append(mean_fitness)

    # Calculate the average and standard deviation for each generation
    avg_max_fitness = [np.mean(fitness) for fitness in max_fitnesses if fitness]  # Filter out empty lists
    avg_mean_fitness = [np.mean(fitness) for fitness in mean_fitnesses if fitness]  # Filter out empty lists
    std_max_fitness = [np.std(fitness) for fitness in max_fitnesses if fitness]  # Filter out empty lists
    std_mean_fitness = [np.std(fitness) for fitness in mean_fitnesses if fitness]  # Filter out empty lists

    # Generate corresponding generations list (without gaps)
    valid_generations = [i for i, fitness in enumerate(max_fitnesses) if fitness]

    return valid_generations, avg_max_fitness, avg_mean_fitness, std_max_fitness, std_mean_fitness

# Function to plot fitness statistics
def plot_fitness_statistics(generations, avg_max_fitness, std_max_fitness, avg_mean_fitness, std_mean_fitness, ea, enemy):
    plt.figure(figsize=(12, 6))

    # Plot mean and max fitness
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
    plt.tight_layout()

    # Save the plot
    output_folder = f"plots/{ea}/{enemy}"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/fitness_statistics_enemy{enemy}.png")
    plt.show()
    plt.close()

# Main function to run the plotting
def main():
    base_data_folder = 'testdata'
    ea = 'EA1'  # Change to the specific EA you want to analyze
    enemy = '2'  # Change to the specific enemy you want to analyze

    ea_folder = os.path.join(base_data_folder, ea)
    enemy_folder = enemy

    # Load the fitness data
    generations, avg_max_fitness, avg_mean_fitness, std_max_fitness, std_mean_fitness = load_fitness_data(ea_folder, enemy_folder)
    plot_fitness_statistics(generations, avg_max_fitness, std_max_fitness, avg_mean_fitness, std_mean_fitness, ea, enemy)
    print(f"Plot saved for EA: {ea}, Enemy: {enemy}")


if __name__ == '__main__':
    main()
