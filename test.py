# standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import itertools

# imports file utils
from file_utils import save_final_best_solution

# Define the folder name where the best solutions are stored
folder_name = 'comp'
file_name = 'best_kills.txt'

# Load the file with numpy
file_path = f'{folder_name}/{file_name}'
weights = np.loadtxt(file_path)

# Save the file with scientific notation and 18 decimal places
np.savetxt(file_path, weights, fmt='%.18e')

