"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This script is used to visualize the results of tuning the
hyperparameters of the EA using Optuna. The script loads the study from the
SQLite database and creates scatter plots for each hyperparameter against the
objective value (mean fitness). The plots have been analysed to determine the
best hyperparameters for the EA based on fitness result.
"""

import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the study from the SQLite database
study_name = "optuna_fitness_max_367"  # Replace with your actual study name
storage_name = "sqlite:///optuna_fitness_max_367.db"  # SQLite database location
# study_name = "optuna_study"  # Replace with your actual study name
# storage_name = "sqlite:///optuna_crowding_mean.db"


method = "HEATMAP"

if method == "HEATMAP":
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    # Convert trials to a pandas DataFrame
    df_trials = study.trials_dataframe()

    # Define the parameters and objective column
    objective_column = 'value'  # The column with the objective value (mean fitness)
    parameters = ['params_population_size', 'params_crossover_rate', 'params_mutation_rate', 'params_mutation_std', 'params_alpha', 'params_tournament_size']

    # Drop rows with NaN or infinite values in any relevant columns
    df_trials = df_trials.replace([np.inf, -np.inf], np.nan).dropna(subset=[objective_column] + parameters)
    print(len(df_trials))
    # Create heatmaps for each parameter vs objective
    plt.figure(figsize=(15, 10))

    for i, param in enumerate(parameters, 1):
        plt.subplot(3, 2, i)
        # Create a 2D histogram (bin the parameter and objective)
        heatmap_data, x_edges, y_edges = np.histogram2d(df_trials[param], df_trials[objective_column], bins=30, density=False)
        
        # Prepare tick labels: every 5th label is shown, the rest are None
        x_labels = np.round(x_edges, 2)
        y_labels = np.round(y_edges, 2)
        
        x_tick_labels = [label if idx % 5 == 0 else None for idx, label in enumerate(x_labels)]
        y_tick_labels = [label if idx % 5 == 0 else None for idx, label in enumerate(y_labels)]
        
        # Plot heatmap
        sns.heatmap(heatmap_data.T, cmap='viridis', cbar=True, 
                    xticklabels=x_tick_labels, yticklabels=y_tick_labels)
        
        # Rotate x-tick labels
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)  # Optional: Change this if you want to rotate y-tick labels as well
        
        plt.title(f'Effect of {param} on Fitness', fontsize = 22)
        # plt.xlabel(param, fontsize = 22)
        plt.ylabel('Fitness', fontsize = 22)

    # plt.tight_layout()
    plt.show()


if method == "SCATTER":
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    # Convert trials to a pandas DataFrame
    df_trials = study.trials_dataframe()

    # Print the first few rows to inspect the data
    # print(df_trials.head())

    # Define the parameters and objective column
    objective_column = 'value'  # The column with the objective value (mean fitness)
    parameters = ['params_population_size', 'params_crossover_rate', 'params_mutation_rate', 'params_mutation_std', 'params_alpha', 'params_tournament_size']

    # Create scatter plots for each parameter vs objective
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(parameters, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(x=df_trials[param], y=df_trials[objective_column])
        plt.title(f'Effect of {param} on Fitness')
        plt.xlabel(param)
        plt.ylabel('Fitness')

    plt.tight_layout()
    plt.show()
