# file_utils.py
import os
import csv
import numpy as np
import time


# Save the best solution weights to a txt file
def save_best_solution(best_solution, best_solution_fitness, enemies, ea_name):
    enemies_name = ''.join(str(e) for e in enemies)

    # Create directory if it doesn't exist
    directory = f"best_solutions/{str(ea_name)}/{enemies_name}/"
    os.makedirs(directory, exist_ok=True)

    # Generate a unique filename using a timestamp to avoid overwriting files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{directory}solution_{timestamp}.txt"

    # Save only the weights as floating-point numbers to the txt file
    np.savetxt(filename, best_solution, fmt='%f')

    print(f"Best solution saved with fitness: {best_solution_fitness} to {filename}")


# Load the best solution weights from a file
def load_best_solution(enemies, ea_name):
    enemies_name = ''.join(str(e) for e in enemies)

    directory = f"best_solutions/{str(ea_name)}/{enemies_name}/"
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


# Save fitness statistics to a CSV file
def save_fitness_stats_to_csv(fitness_stats, enemies, ea_name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    enemies_name = ''.join(str(e) for e in enemies)

    # Define the path for the CSV file
    csv_path = f"testdata/{ea_name}/{enemies_name}"
    os.makedirs(csv_path, exist_ok=True)
    csv_filename = f"{csv_path}/fitness_stats_enemy{enemies_name}_{timestamp}.csv"

    # Write the fitness statistics to a CSV file
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "Max Fitness", "Mean Fitness", "Std Fitness", "Variety"])
        writer.writerows(fitness_stats)

    print(f"Fitness statistics saved to {csv_filename}")


# save the best final solution including individual gain
def save_final_best_solution(best_solution, individual_gain, enemies, ea_name, name=None):
    enemies_name = ''.join(str(e) for e in enemies)

    # Create directory if it doesn't exist
    directory = f"final_best_solutions/{str(ea_name)}/{enemies_name}/"
    os.makedirs(directory, exist_ok=True)

    # Generate a unique filename using a timestamp to avoid overwriting files
    if not name:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{directory}solution_{timestamp}.txt"
    else:
        filename = f"{directory}{name}.txt"

    # Save the individual gain along with the weights
    best_solution = np.append(individual_gain, best_solution)
    np.savetxt(filename, best_solution, fmt='%f')
