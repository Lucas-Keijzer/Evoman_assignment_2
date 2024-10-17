from file_utils import load_best_solutions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_table_and_heatmaps(df, ea, best_enemies_beaten, enemy_group):
    """
    Function to plot the table and the heatmap.
    """
    # Create a new figure for the table and heatmap
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    # 1. Display the table
    axs[0].axis('off')  # Hide the axes for the table
    table_data = df.values
    column_labels = df.columns

    # Create table within the Matplotlib plot
    table = axs[0].table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale to fit

    # 2. Plot the heatmap for the best enemies beaten
    sns.heatmap(best_enemies_beaten.reshape(1, -1), cmap='RdYlGn', annot=True, cbar=True,
                vmin=-1, vmax=100, linewidths=1, linecolor='black', fmt='.0f', ax=axs[1])
    axs[1].set_title(f"Best Enemies Beaten for EA: {ea}, Enemy Group: {enemy_group}")
    axs[1].set_xlabel("Enemy Index")
    axs[1].set_yticks([])  # Hide y-axis ticks since it's just one row

    # Adjust layout to fit the table and heatmap
    plt.tight_layout()
    plt.show()


def main():
    # Example enemy groups (replace with all 56 enemy groups)
    enemy_groups = [[1, 2, 3], [4, 5, 6]]  # Add all 56 enemy groups here

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
            enemies_beaten_std = np.std(number_of_enemies_beatens)

            # Identify the best solution (max gain)
            best_solution_index = np.argmax(gains)
            best_enemies_beaten = enemies_beatens[best_solution_index]

            # Append stats to the data_for_table list
            data_for_table.append({
                'EA': ea,
                'Enemy Group': str(enemy_group),
                'Gain Mean ± Std': f"{gains_mean:.2f} ± {gains_std:.2f}",
                'Gain Max': gains_max,
                'Enemies Beaten Mean ± Std': f"{enemies_beaten_mean:.2f} ± {enemies_beaten_std:.2f}",
                'Enemies Beaten Max': enemies_beaten_max
            })

            # After each group, convert data to DataFrame and plot
            df = pd.DataFrame(data_for_table)
            plot_table_and_heatmaps(df, ea, best_enemies_beaten, enemy_group)


if __name__ == '__main__':
    main()
