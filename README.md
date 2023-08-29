# Private Causal Neural Networks.
In this repository, we open-source the (private) causal neural network from our paper. This library allows you to estimate the CATE of any targeting intervention while preserving the privacy of your subjects in a meaningful way.

## causal_neural_network function
We provide a function that replicates our estimator of the CATE in causal_neural_network.py. This function requires the following parameters: 

1. `X`: a numpy set of covariates.
2. `Y`: a numpy set of outcome variable (e.g., revenue).
3. `T`: a numpy vector of the treatment received.
4. `scaling`: requires whether `X` should be scaled (boolean). We advice to use scaling, it makes the estimator more precise.
5. `simulations`: requires the number of average CATE's (int).
6. `epochs`: requires the number of epochs (int).
7. `batch_size`: requires the batch size (int).
8. `folds`: requires the number of folds for K-fold cross-validation.

It returns:

1. the average treatment effect in `average_treatment_effect`.
2. the uplift or CATE per customer in `CATE_estimates`.
3. a Keras model that predicts the CATE based on covariates X in `tau_hat`.

## Example using simulated data.
