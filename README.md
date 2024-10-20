This folder contains some python and data files for the creation of two different
EAs. These EAs are tested against one another based on some different statistics.

Folders:
  - final_best_solutions:
      Contains 2 * 56 * 10 best solutions in txt format. These are all the best
      solutions found by both EAs for all the different combinations of 3
      enemies and then the best of 10 runs for each of these combinations.

  - final_data:
      Contains 10 csv files for 10 different training runs. Per EA 10 runs have
      been done per enemy group, which means 2 * 10 * 2 = 40 runs/files.

Files:
  - run_best_solutions.py:
      Contains the implementation to load any of the best solutions and run them
      in the environment to see how they perform. Visuals and data included if
      you're interested ;). Running this file can be done:

      'python3/python run_best_solutions.py'


  - fitness_based.py:
      Contains the implementation of 'EA1' which makes use of fitness based
      generation replacement. Running this file can be done using:

      'python3/python fitness_sharing.py

      This reruns the experiments and stores them in a new folder 'testdata'.
      NOTE: this will take about an hour to run.


  - age_based.py:
      Contains the implementation of 'EA2' which makes use of age based
      generation replacement. Running this file can be done using:

      'python3/python crowding.py'

      This reruns the experiments and stores them in a new folder 'testdata'.
      NOTE: this will take about an hour to run.

  - plot.py:
      Contains the implementation to plot the data for the fitness across the
      generations. It also plots the diversity and the individual gain for all
      the best solutions in a box plots and lastly shows the statistical test
      for the results of the different setups.
      l Running this file can be done using:

      'python3/python plot.py'

      This generates and shows the plots.

  - get_best_individuals.py:
      Contains the implementation to load the best solutions and get the best
      individual from each of the runs. It does so by looking at all the best
      solutions for a given setup and extracts the 10 best individuals from all
      the runs. Used to get the boxplot data.
      Running this file can be done using:

      'python3/python get_best_individuals.py'

  - #.db:
      the optuna databases for the different setups. These are used to store the
      results of the hyperparameter optimization for the different setups.

  - parameter_plots.py:
      Contains the implementation to plot the results of the hyperparameter
      optimization. Running this file can be done using:

      'python3/python parameter_plots.py'
