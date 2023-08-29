# Private Causal Neural Networks.
In this repository, we open-source the (private) causal neural network from our paper. This library allows you to estimate the CATE of any targeting intervention while preserving the privacy of your subjects in a meaningful way.

## causal_neural_network.py
We provide a function that replicates our estimator of the CATE in causal_neural_network.py. This function requires the following parameters: 

1. `X`: a numpy set of covariates.
2. `Y`: a numpy set of outcome variable (e.g., revenue).
3. `T`: a numpy vector of the treatment received.
4. `scaling`: requires whether `X` should be scaled (boolean). We advice to use scaling, it makes the estimator more precise.
5. `simulations`: requires the number of average CATE's (int).
6. `epochs`: requires the number of epochs (int).
7. `max_epochs`: requires the number of maximum epochs for hyperparameter tuning (int).
8. `batch_size`: requires the batch size (int).
9. `folds`: requires the number of folds for K-fold cross-validation.
10. `directory`: folder drive to save the hyperparameter tuning results.

Returns:

1. `average_treatment_effect`: the average treatment effect.
2. `CATE_estimates`: the uplift or CATE per customer.
3. `tau_hat`: a Keras model that predicts the CATE based on covariates X.

## private_causal_neural_network.py
We provide a function that replicates our estimator of the CATE in private_causal_neural_network.py. This function requires the following parameters: 

1. to 10. same as the above.
11. `noise_multiplier`: parameter to control the amount of privacy risk you are willing to sacrifice.
12. `l2_norm_clip`: parameter to control the sensitivity of the gradient (default = 4).

Returns: 
1. `average_treatment_effect`: the average treatment effect.
2. `CATE_estimates`: the uplift or CATE per customer.
3. `tau_hat`: a Keras model that predicts the CATE based on covariates X.
4. `epsilon`: the privacy risk level $\varepsilon$.

## Example using simulated data.
