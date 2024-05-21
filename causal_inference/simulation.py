# install PCNNs and simulation setup
#!pip install git+https://github.com/GilianPonte/PrivateTargeting.git -q

import os
import pandas as pd
import numpy as np
import random
import tensorflow
import simulation_data
import strategy1
import time


def set_seed(seed):
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)
    tensorflow.keras.utils.set_random_seed(seed)


def main(datafile, noise_multiplier, iterations, seed):
    # try:
    #     seeds_data = np.genfromtxt('/content/seeds_data.txt', delimiter=',', dtype=np.int64)
    # except IOError:
    #     print("Error: File not found or could not be read.")
    # try:
    #     seeds_training = np.genfromtxt('/content/seeds_training.txt', delimiter=',', dtype=np.int64)
    # except IOError:
    #     print("Error: File not found or could not be read.")

    # set time
    start_time = time.time()
    tensorflow.config.experimental.enable_op_determinism()
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Read seeds_data and seeds_training from file
    # seeds_data = read_file("seeds_data.txt")
    # seeds_training = read_file("seeds_training.txt")

    # simulation parameters
    results_list = []
    #noise_multipliers = [0] # Initialize lists to store results for each noise multiplier ,8.7,3.4,1.12,0.845,0.567,0.3543

    set_seed(seed)

    # read data
    try:
        data = pd.read_csv(datafile)
    except FileNotFoundError:
        data = simulation_data.data_simulation(100_000)
        data.to_csv(datafile)
        set_seed(seed)  # Reset seed

    for i in range(iterations):

        # Separate the columns into different DataFrames
        x = data[['covariate_1', 'covariate_2', 'covariate_3', 'covariate_4', 'covariate_5', 'covariate_6']]
        w, m, tau, mu1, mu0, y = data[['w']], data[['m']], data[['tau']], data[['mu1']], data[['mu0']], data[['y']]

        # Define the directory based on the noise multiplier
        directory = f"tuner_{noise_multiplier}_iteration_algo_run"
        os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

        # Call the function with the current noise_multiplier value
        if noise_multiplier == 0:
          print("no privacy")
          average_treatment_effect, CATE_estimates, tau_hat = strategy1.cnn(X = x,
                                                                    Y = y,
                                                                    T = w,
                                                                    scaling = True,
                                                                    batch_size = 100,
                                                                    epochs = 100,
                                                                    max_epochs = 10,
                                                                    folds = 10,
                                                                    directory = directory)
          epsilon = 0
          n = len(x)
          epsilon_conservative = 0
        if noise_multiplier != 0:
          average_treatment_effect, CATE_estimates, tau_hat, n, epsilon, noise_multiplier, epsilon_conservative = strategy1.pcnn(
              X=x,
              Y=y,
              T=w,
              scaling=True,
              batch_size=100,
              epochs=100,
              max_epochs=10,
              fixed_model = True,
              directory=directory,  # Use the directory variable here
              noise_multiplier=noise_multiplier)
        # Append the results to the list
        results_list.append({
            'Noise Multiplier': noise_multiplier,
            'Average Treatment Effect': average_treatment_effect,
            'CATE Estimates': CATE_estimates,
            'Epsilon': epsilon,
            'epsilon conservative': epsilon_conservative,
            'true ate': data[['tau']].mean(),
            'true CATE': data[['tau']],
            'sample size': n,
            'data set': i,
            #'iteration': a,
            'covariates' : x,
            'y': y,
            'w': w,
          })

        np.save(f"results_list_{noise_multiplier}", results_list)

        # Print or use the DataFrames as needed
        print(results_list)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time one sim: {:.2f} seconds".format(execution_time))



if __name__ == '__main__':

    import sys

    assert len(sys.argv) == 4
    noise_multiplier = float(sys.argv[1])
    iterations = int(sys.argv[2])
    seed = int(sys.argv[3])

    main('simdata.csv', noise_multiplier=noise_multiplier, iterations=iterations, seed=seed)
