import pandas as pd
import numpy as np

def protect_selection(epsilon, selection, top):
    # Privacy settings
    exp_epsilon = np.exp(epsilon)
    P = np.array([[exp_epsilon, 1 - exp_epsilon], 
                  [1 - exp_epsilon, exp_epsilon]]) / (1 + exp_epsilon)
    
    # Get responses
    responses = np.zeros(len(selection), dtype=int)
    
    # Generate protected selection based on matrix above
    for i in range(len(selection)):
        responses[i] = np.random.choice([0, 1], p=P[1, :] if selection[i] == 1 else P[0, :])
    
    # Initialize protected selection
    protected_selection = np.zeros(len(selection), dtype=int)
    
    # Select indices of zeros and ones
    index_0, index_1 = responses == 0, responses == 1
    
    # Apply selection logic
    if top > np.sum(index_1):
        protected_selection[np.random.choice(np.where(index_1)[0], np.sum(index_1), replace=False)] = 1
        protected_selection[np.random.choice(np.where(index_0)[0], top - np.sum(index_1), replace=False)] = 1
    else:
        protected_selection[np.random.choice(np.where(index_1)[0], top, replace=False)] = 1
    
    return protected_selection

def protect_CATEs(percent, CATE, CATE_estimates, n, epsilons=[0.05, 0.5, 1, 3, 5]):
    top = int(np.floor(n * percent))
    selection_true = np.zeros(n, dtype=int)
    selection_tau = np.zeros(n, dtype=int)
    
    # Determine top selections based on CATE estimates
    selection_tau[np.argsort(-CATE_estimates)[:top]] = 1
    if len(CATE) > 0:
        selection_true[np.argsort(-CATE)[:top]] = 1
    
    # Now with local dp
    collection = pd.DataFrame({'customer': np.arange(1, n+1)})
    for epsilon in epsilons:
        print(epsilon)
        protected_selection = protect_selection(epsilon, selection_tau, top)
        collection[f"epsilon_{str(epsilon).replace('.', '')}"] = protected_selection
    
    collection['random'] = np.random.choice([0, 1], size=n, p=[1-percent, percent])
    collection['percentage'] = percent
    collection['selection_true'] = selection_true
    collection['selection_tau'] = selection_tau
    if len(CATE) > 0:
        collection['tau'] = CATE
    
    return collection

def bootstrap_strat_2(bootstraps, CATE, CATE_estimates, percentage):
    # Initialize a DataFrame to store bootstrap results
    bootstrap_results = pd.DataFrame()
    
    # Loop over each bootstrap iteration
    for b in range(bootstraps):
        print(b)
        # Resample the data with replacement
        bootstrap_data = np.random.choice(CATE, size=len(CATE), replace=True)
        
        # Initialize an object to store results for this bootstrap
        percentage_collection = pd.DataFrame()
        
        # Loop over each percentage level
        for percent in percentage:
            collection = protect_CATEs(percent=percent, CATE=bootstrap_data, CATE_estimates=CATE_estimates, n=len(CATE_estimates), epsilons=[0.05, 0.5, 1, 3, 5])
            collection['percent'] = percent
            percentage_collection = pd.concat([percentage_collection, collection], ignore_index=True)
        
        # Store the results from this bootstrap iteration
        percentage_collection['bootstrap'] = b
        bootstrap_results = pd.concat([bootstrap_results, percentage_collection], ignore_index=True)
    
    return bootstrap_results
