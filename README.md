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
  - fitness_based.py:
      Contains the implementation of 'EA1' which makes use of fitness based
      generation replacement. Running this file can be done using:

      'python3/python fitness_sharing.py

      This reruns the experiments and stores them in a new folder 'testdata'.
      NOTE: this will take about an hour to run.


  - crowding.py:
      Contains the implementation of 'EA2' which makes use of crowding, a
      crossover technique to preserve diversity.
      Running this file can be done using:

      'python3/python crowding.py'

      This reruns the experiments and stores them in a new folder 'testdata'.
      NOTE: this will take about an hour to run.

  - plot.py:
      Contains the implementation to plot the data from the csv files in the
      'final_data' folder. Running this file can be done using:

      'python3/python plot.py'

      This generates the plots and stores them in the 'plots' folder.

  - run_best_solutions.py:
      Contains the implementation to run the best solutions for both EAs against
      the three different enemies. It also generates the boxplots for these runs.
      Running this file can be done using:

      'python3/python run_best_solutions.py'

      This generates the boxplot and stores the it in the plots folder. It also
      prints the results of the statistical tests in the terminal to see if the
      two EAs significantly differ from one another.
