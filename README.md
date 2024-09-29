This folder contains some python and data files for the creation of two different
EAs. These EAs are tested against one another based on some different statistics.

Folders:
  - final_best_solutions:
      Contains 6 best solutions: one per enemy for three different enemies for
      both EAs

  - final_data:
      Contains 10 csv files for 10 different training runs. Per EA 10 runs have
      been done per enemy, which means 2 * 3 * 10 = 60 runs.

  - plots:
      Contains the plots generated to show the data between the two different
      EAs.

  - tuning:
      Contains all the files used and generated for the hyperparameter tuning
      of both of the EAs.

Files:
  - fitness_sharing.py:
      Contains the implementation of 'EA1' which makes use of fitness sharing
      to preserve the diversity. Running this file can be done using:

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
