import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the study from the SQLite database
study_name = "optuna_crowding_max"  # Replace with your actual study name
storage_name = "sqlite:///optuna_crowding_max.db"  # SQLite database location
# study_name = "optuna_study"  # Replace with your actual study name
# storage_name = "sqlite:///optuna_crowding_mean.db"


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
