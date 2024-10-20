"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle


"""


from file_utils import load_best_solutions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools


# prints the table of all the results in the terminal
def print_table_in_terminal(df):
    # Use pandas to_string method to print the DataFrame as a formatted table in the terminal
    print(df.to_string(index=False))


def main():
    # Example enemy groups (replace with all 56 enemy groups)
    enemy_groups = list(itertools.combinations(range(1, 9), 3))

    # Prepare a list to store statistics for the table
    data_for_table = []

    for ea in ['EA1', 'EA2']:
        for enemy_group in enemy_groups:
            gains = []
            number_of_enemies_beatens = []
            enemies_beatens = []

            # Collect data
            for individual_gain, number_of_enemies_beaten, enemies_beaten, _ in load_best_solutions(ea, enemy_group):
                gains.append(individual_gain)
                number_of_enemies_beatens.append(number_of_enemies_beaten)
                enemies_beatens.append(enemies_beaten)

            # Convert lists to arrays for easier calculations
            gains = np.array(gains)
            number_of_enemies_beatens = np.array(number_of_enemies_beatens)
            enemies_beatens = np.array(enemies_beatens)  # Shape (10, 8)

            # Calculations
            gains_mean = np.mean(gains)
            gains_max = np.max(gains)
            gains_std = np.std(gains)

            enemies_beaten_mean = np.mean(number_of_enemies_beatens)
            enemies_beaten_max = np.max(number_of_enemies_beatens)
            enemies_beaten_std = np.std(number_of_enemies_beaten)

            # Identify the best solution (max gain)
            best_solution_index = np.argmax(gains)
            best_enemies_beaten = enemies_beatens[best_solution_index]

            # Prepare data for the table
            # Get indices of enemies beaten with player lives > -1
            indices_of_enemies_beaten = [i+1 for i, life in enumerate(best_enemies_beaten) if life > -1]
            if enemy_group == (4,6,8):
                print(indices_of_enemies_beaten)
                print(best_enemies_beaten)
            # Join the indices as a comma-separated string
            total_enemies_beaten = ', '.join(map(str, indices_of_enemies_beaten))

            # Append stats and enemies beaten to the data_for_table list
            data_for_table.append({
                'EA': ea,
                'Enemy Group': str(enemy_group),
                'Gain Mean ± Std': f"{gains_mean:.2f} ± {gains_std:.2f}",
                'Gain Max': gains_max,
                '# kills Mean ± Std': f"{enemies_beaten_mean:.2f} ± {enemies_beaten_std:.2f}",
                'Max # kills': enemies_beaten_max,
                'Enemies Beaten': total_enemies_beaten  # Last column with total enemies beaten
            })

            # After each group, convert data to DataFrame and print it
            df = pd.DataFrame(data_for_table)

        # Print the table to the terminal
        print_table_in_terminal(df)


if __name__ == "__main__":
    main()
