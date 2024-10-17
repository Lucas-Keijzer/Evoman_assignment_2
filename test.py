"""
Authors: Lucas Keijzer, Pjotr Piet, Max Scot, Marina Steinkuhle

Description: This file loads all the best individuals from all the 'n'
experiments done by either EA, and finds the 10 best individuals based on
the 'individual gain' values. It also finds the two best individuals based on
the amount of enemies beaten and one on the individual gain for the competition.
These best few individuals are then saved in a separate folder for further
analysis and comparison.
"""

# standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import itertools




