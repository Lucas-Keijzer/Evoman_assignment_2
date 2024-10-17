from file_utils import load_best_solutions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_table_and_heatmap_in_row(df, ea, best_enemies_beaten, enemy_group):
    """
    Function to plot the table and the heatmap in the same row.
    """
    # Increase the figure width to provide enough space
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjusted width and height for table

    # Turn off the axis for the table
    ax.axis('off')

    # Create the table
    table_data = df.values
    column_labels = df.columns

    # Create table within the Matplotlib plot
    table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale to fit

    # Display the figure
    plt.show()


def main():
    # Example enemy groups (replace with all 56 enemy groups)
    enemy_groups = [[1, 2, 3]]  # Add all 56 enemy groups here

    # Prepare a list to store statistics for the table
    data_for_table = []

    for ea in ['EA2']:
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

            # After each group, convert data to DataFrame and plot
            df = pd.DataFrame(data_for_table)

            # Plot the table and heatmap
            plot_table_and_heatmap_in_row(df, ea, best_enemies_beaten, enemy_group)


if __name__ == "__main__":
    main()
