import numpy as np
import pandas as pd

def protect_CATEs(percent, CATE, CATE_estimates, n, epsilons=[0.05, 0.5, 1, 3, 5], seed=1):
    np.random.seed(seed)
    top = int(n * percent)
    selection_true = np.zeros(n)
    selection_tau = np.zeros(n)
    indices_tau = np.argsort(CATE_estimates)[::-1][:top]
    selection_tau[indices_tau] = 1
    if len(CATE) > 0:
        indices_true = np.argsort(CATE)[::-1][:top]
        selection_true[indices_true] = 1

    collection = pd.DataFrame({'customer': np.arange(1, n+1)})
    for epsilon in epsilons:
        protected_selection = protect_selection(epsilon, selection_tau, top)
        collection[f'epsilon_{epsilon:.2f}'.replace('.', '')] = protected_selection
    
    collection['random'] = np.random.choice([0, 1], size=n, replace=True, p=[1-percent, percent])
    collection['percentage'] = percent
    collection['selection_true'] = selection_true
    collection['selection_tau'] = selection_tau
    if len(CATE) > 0:
        collection['tau'] = CATE
    return collection

def protect_selection(epsilon, selection, top, seed=1):
    np.random.seed(seed)
    P = np.zeros((2, 2))
    exp_eps = np.exp(epsilon)
    P[np.diag_indices_from(P)] = exp_eps / (2 - 1 + exp_eps)
    P[P == 0] = 1 / (2 - 1 + exp_eps)
    responses = np.zeros(len(selection))
    for i in range(len(selection)):
        np.random.seed(seed + i)
        responses[i] = np.random.choice([0, 1], p=P[int(selection[i]), :])
    protected_selection = np.zeros(len(selection))
    index_0 = np.where(responses == 0)[0]
    index_1 = np.where(responses == 1)[0]
    if top > len(index_1):
        protected_selection[np.random.choice(index_1, len(index_1), replace=False)] = 1
        protected_selection[np.random.choice(index_0, top - len(index_1), replace=False)] = 1
    else:
        protected_selection[np.random.choice(index_1, top, replace=False)] = 1
    return protected_selection
    
def bootstrap_strat_2(bootstraps, CATE, CATE_estimates, percentage=np.arange(0.1, 1, 0.1), epsilons=[0.05, 0.5, 1, 3, 5], seed=1):
    np.random.seed(seed)
    seeds = np.random.choice(range(1, 1000000), size=bootstraps, replace=False)
    bootstrap_results = pd.DataFrame()
    for b in range(bootstraps):
        np.random.seed(seeds[b])
        bootstrap_data = np.random.choice(CATE, size=len(CATE), replace=True)
        percentage_collection = pd.DataFrame()
        for percent in percentage:
            np.random.seed(seeds[b])
            collection = protect_CATEs(percent, bootstrap_data, CATE_estimates, len(CATE_estimates), epsilons, seeds[b])
            collection['percent'] = percent
            percentage_collection = pd.concat([percentage_collection, collection], ignore_index=True)
        percentage_collection['bootstrap'] = b
        bootstrap_results = pd.concat([bootstrap_results, percentage_collection], ignore_index=True)
    return bootstrap_results
