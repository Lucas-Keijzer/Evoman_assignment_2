"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This script is used to execute the best found solutions during
the 10 runs of the two different EAs. The best solutions are saved in the
final_best_solutions folder and are loaded from there. The best solutions are
played against the respective three different enemies with RANDOM INITIALISATION
(might change this back) and the results are visualized in boxplots.
Additionally, a statistical test is performed to determine if the results are
significantly different.
"""

# imports framework
import os
import numpy as np
import matplotlib.pyplot as plt
from evoman.environment import Environment
from demo_controller import player_controller
import scipy.stats as stats
import pandas as pd

experiment_name = 'boxplots'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific exampleload_best_solution
n_hidden_neurons = 10

folder_name = 'final_best_solutions'


def plot_boxplots(gains_ea1, gains_ea2, enemies):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharey=True)  # sharey=True for shared y-axis

    for i, enemy in enumerate(enemies):
        # Prepare data for boxplot
        data = [gains_ea1[enemy], gains_ea2[enemy]]

        # Create the boxplot
        axes[i].boxplot(data, labels=['EA1', 'EA2'])

        # Increase font sizes
        axes[i].set_title(f'Enemy {enemy}', fontsize=28)  # Increase title size
        axes[i].tick_params(axis='y', labelsize=12)  # Increase y-axis tick label size
        axes[i].tick_params(axis='x', labelsize=12)  # Increase x-axis tick label size

    # Set shared y-axis label, placing it outside of the plot
    fig.text(0.03, 0.5, 'Gain (Player Life - Enemy Life)', va='center', rotation='vertical', fontsize=24)

    # Add a title for the entire figure
    fig.suptitle('Boxplots of 5 Runs per EA per Enemy', fontsize=30)

    # Adjust layout to prevent overlap and leave space for the y-label and title
    plt.subplots_adjust(left=0.06, right=0.95, top=0.85, bottom=0.1)  # Increased top to fit the title
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Ensure layout is adjusted for y-label and title

    # Save the plot in the existing 'plots' folder as 'boxplot.png'
    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/boxplot.png")

    plt.show()
    plt.close()


def statistical_test(gains_ea1, gains_ea2, enemies):
    results = []

    for enemy in enemies:
        # Gather data for each enemy
        data_ea1 = gains_ea1[enemy]
        data_ea2 = gains_ea2[enemy]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data_ea1, data_ea2)

        # Calculate means and 95% confidence intervals
        mean_ea1 = np.mean(data_ea1)
        mean_ea2 = np.mean(data_ea2)

        ci_ea1 = stats.t.interval(0.95, len(data_ea1)-1, loc=mean_ea1, scale=stats.sem(data_ea1))
        ci_ea2 = stats.t.interval(0.95, len(data_ea2)-1, loc=mean_ea2, scale=stats.sem(data_ea2))

        results.append({
            'Enemy': enemy,
            'Mean EA1': mean_ea1,
            'Mean EA2': mean_ea2,
            'T-Statistic': t_stat,
            'P-Value': p_value,
            '95% CI EA1': ci_ea1,
            '95% CI EA2': ci_ea2,
            'Significant Difference': p_value < 0.05  # True if significant
        })

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)

    return results_df


def main():
    trained_enemies = [2, 5, 8]

    # enemies_folder = ''.join([str(enemy) for enemy in trained_enemies])
    all_enemies = range(1, 9) # List of enemies to test agianst

    enemy_groups = [[1, 2, 3, 4, 5, 6, 7, 8]]
    enemy_groups = [[3, 6, 7]]

    # goated agents:EA2:
    # 125 4,8,9
    # 126 0,
    # 138 5,6
    # 145 5,
    # 234 1,6
    # 247 2(5!),3(5!),9
    # 248 0,5,6,7,9
    # 256 2,3,4(5!),9
    # 257 4(5!+180),7,8,
    # 378 5,7,9
    # 457 2,8,9
    # 458 -(miss 7 toevoegen voor generalisatie)


    # for ea in ['EA1', 'EA2']:
    for ea in ['EA1']:
        gainss = []
        for enemy in enemy_groups:
            enemies_folder = ''.join([str(enemy) for enemy in enemy])
            gains = []
            lives = []
            for enemy in all_enemies:

                # Load the weights from the file
                # folder_path = f'{folder_name}/{ea}/{enemies_folder}'
                # file_name = os.listdir(folder_path)
                # file_path = os.path.join(folder_path, file_name)

                # weights = np.loadtxt(file_path)
                weights = [0.155619,
                    0.496913,
                    -0.357366,
                    0.491793,
                    -0.642359,
                    -0.294789,
                    -0.435402,
                    0.205539,
                    -0.417421,
                    0.583039,
                    -0.932929,
                    -1.000000,
                    0.837966,
                    0.229272,
                    -0.380106,
                    0.898680,
                    0.854255,
                    0.754916,
                    0.308799,
                    0.981079,
                    -0.352229,
                    -0.788346,
                    0.793919,
                    -1.000000,
                    0.442789,
                    0.263389,
                    -0.320588,
                    0.384091,
                    0.602881,
                    -0.172602,
                    -0.343957,
                    0.815579,
                    -1.000000,
                    1.000000,
                    -0.835791,
                    0.822503,
                    1.000000,
                    0.990020,
                    0.113019,
                    -0.228178,
                    -0.943200,
                    0.732798,
                    0.813503,
                    -0.003531,
                    -0.589215,
                    0.492930,
                    -0.132208,
                    -0.284249,
                    0.719372,
                    -0.848691,
                    -0.748800,
                    -0.573512,
                    0.482176,
                    0.997059,
                    0.206800,
                    0.287431,
                    -0.046293,
                    0.283612,
                    -0.103346,
                    0.403499,
                    -0.895924,
                    0.571823,
                    0.965164,
                    -0.539707,
                    -0.139888,
                    -0.971202,
                    0.039265,
                    0.481591,
                    -0.560235,
                    -0.950849,
                    0.355245,
                    -1.000000,
                    -0.786933,
                    0.931002,
                    0.205801,
                    -0.535602,
                    -0.402233,
                    -0.296651,
                    0.064214,
                    0.069918,
                    0.055423,
                    0.268623,
                    -0.977367,
                    0.266562,
                    -0.897245,
                    0.532588,
                    0.069490,
                    0.801281,
                    0.839709,
                    -0.182415,
                    0.554475,
                    0.270886,
                    -0.676825,
                    -0.925854,
                    -0.029511,
                    -0.014965,
                    1.000000,
                    0.527632,
                    -0.678536,
                    -0.245787,
                    -0.175871,
                    -0.324970,
                    1.000000,
                    0.104612,
                    0.102230,
                    -0.674096,
                    0.007753,
                    -0.169557,
                    -0.815472,
                    0.274162,
                    -0.933935,
                    0.107705,
                    0.618762,
                    0.200140,
                    0.953001,
                    -1.000000,
                    -0.039263,
                    0.976633,
                    0.067261,
                    -0.583597,
                    0.296177,
                    0.026831,
                    0.080192,
                    -0.420483,
                    -0.191783,
                    -0.397212,
                    0.073308,
                    0.419549,
                    0.073329,
                    0.169513,
                    -0.186941,
                    0.750342,
                    -0.462072,
                    0.877956,
                    1.000000,
                    -0.234187,
                    0.440549,
                    0.425385,
                    -0.857987,
                    -0.758901,
                    -0.431095,
                    -0.971309,
                    -1.000000,
                    -0.204841,
                    -1.000000,
                    0.216298,
                    0.392237,
                    0.251116,
                    -0.849837,
                    0.677907,
                    0.286809,
                    0.584711,
                    -0.132114,
                    -0.897470,
                    0.921642,
                    -0.373176,
                    0.860972,
                    -0.022321,
                    -0.235134,
                    0.350992,
                    0.599265,
                    -0.819754,
                    0.882775,
                    0.480672,
                    -0.473078,
                    0.030720,
                    0.670240,
                    0.776193,
                    0.697270,
                    -0.918341,
                    -0.508748,
                    0.749309,
                    0.722839,
                    -0.066879,
                    0.996759,
                    0.739658,
                    -0.060884,
                    0.262571,
                    -0.457495,
                    -0.539965,
                    -0.013721,
                    0.249206,
                    0.796726,
                    0.951065,
                    -0.245069,
                    0.395830,
                    0.295618,
                    -0.547474,
                    0.029413,
                    0.305885,
                    0.142466,
                    0.983982,
                    -0.070159,
                    0.583479,
                    0.175583,
                    -0.949855,
                    -0.268602,
                    0.655067,
                    -0.429941,
                    0.666029,
                    0.525172,
                    -0.290460,
                    -0.387536,
                    -0.231981,
                    0.945809,
                    0.018143,
                    0.673407,
                    0.556852,
                    -0.632396,
                    -0.689359,
                    0.036004,
                    0.010429,
                    0.889761,
                    0.280403,
                    0.583358,
                    0.139776,
                    -1.000000,
                    0.664243,
                    -0.802208,
                    -0.164216,
                    -0.295926,
                    0.907930,
                    -0.985703,
                    0.659107,
                    0.815822,
                    0.251711,
                    -0.667609,
                    0.211097,
                    0.218087,
                    -0.775621,
                    -0.956805,
                    0.120344,
                    0.289324,
                    -0.180001,
                    -0.463911,
                    -0.736658,
                    0.494186,
                    -0.058082,
                    -0.939265,
                    -0.600235,
                    -0.149489,
                    0.705989,
                    -0.382500,
                    0.733573,
                    -0.847058,
                    -0.025952,
                    -0.632061,
                    0.185324,
                    0.391562,
                    0.276654,
                    -0.775325,
                    -0.345559,
                    -0.281604,
                    0.998010,
                    0.196110,
                    -0.086326,
                    -0.192712,
                    -1.000000,
                    0.012105,
                    -0.052157,
                    0.007246,
                    0.150030,
                    0.622761,
                    0.801875,
                    -0.482834]

                weights = np.array(weights)

                # Initializes environment for single objective mode (specialist) with static enemy and AI player
                env = Environment(experiment_name=experiment_name,
                                playermode="ai",
                                enemies=[enemy],
                                player_controller=player_controller(n_hidden_neurons),
                                speed="normal",
                                enemymode="static",
                                level=2,
                                visuals=True)

                # Play and get the results
                fitness, player_life, enemy_life, _ = env.play(weights)

                # Calculate gain for this enemy
                gain = player_life - enemy_life
                gains.append(gain)
                lives.append((player_life, enemy_life))

                print(f"EA {ea}, Enemy {enemy}, Player Life: {player_life}, Enemy Life: {enemy_life}, Gain: {gain}")
            gainss.append(gains)

    print(gainss)
    for i, gains in enumerate(gainss):
        print({i + 1})
        print(gainss)
        print(f'EA1 total gain agianst all enemies: = {sum(gains)}')
        print(f'EA1 beat enemies: {[(i + 1, pl) for i, (pl, el) in enumerate(lives) if el <=0]}')


if __name__ == '__main__':
    main()
