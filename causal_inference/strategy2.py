import numpy as np
import pandas as pd

def protect_CATEs(percent, CATE, CATE_estimates, n, epsilons=[0.05, 0.5, 1, 3, 5], seed=1):
    np.random.seed(seed)
    top = int(n * percent)
    indices_tau = np.argsort(CATE_estimates)[-top:]
    selection_tau = np.zeros(n)
    selection_tau[indices_tau] = 1

    selection_true = np.zeros(n)
    if len(CATE) > 0:
        indices_true = np.argsort(CATE)[-top:]
        selection_true[indices_true] = 1

    # Pre-allocate data for DataFrame construction
    data = {
        'customer': np.arange(1, n+1),
        'random': np.random.choice([0, 1], size=n, replace=True, p=[1-percent, percent]),
        'percentage': np.full(n, percent),
        'selection_true': selection_true,
        'selection_tau': selection_tau
    }
    
    # Precompute values for each epsilon
    for epsilon in epsilons:
        data[f'epsilon_{epsilon:.2f}'.replace('.', '')] = protect_selection(epsilon, selection_tau, top)

    if len(CATE) > 0:
        data['tau'] = CATE

    return pd.DataFrame(data)

def protect_selection(epsilon, selection, top, seed=1):
    np.random.seed(seed)
    exp_eps = np.exp(epsilon)
    p_stay = exp_eps / (1 + exp_eps)
    p_switch = 1 - p_stay

    # Create a probability matrix where:
    # P[0, :] is the probability row for selection == 0
    # P[1, :] is the probability row for selection == 1
    P = np.array([[p_switch, p_stay],   # if not selected, mostly stay not selected
                  [p_stay, p_switch]])  # if selected, mostly stay selected

    # Vectorized random choice based on selection indices
    # This utilizes the fact that P is structured to directly map to the selection array
    responses = np.array([np.random.choice([0, 1], p=P[int(sel)]) for sel in selection])

    # Efficiently finding indices where responses are 1 and making the top selections
    indices = np.where(responses == 1)[0]
    if len(indices) < top:
        protected_indices = indices  # Not enough indices to fill 'top', use all
    else:
        protected_indices = np.random.choice(indices, size=top, replace=False)

    # Create the final protected selection array
    protected_selection = np.zeros_like(selection)
    protected_selection[protected_indices] = 1

    return protected_selection

def bootstrap_strat_2(bootstraps, CATE, CATE_estimates, percentage=np.arange(0.05, 1, 0.05), epsilons=[0.05, 0.5, 1, 3, 5], seed=1):
    np.random.seed(seed)
    seeds = np.random.choice(range(1, 1000000), size=bootstraps, replace=False)
    bootstrap_results = pd.DataFrame()
    for b in range(bootstraps):
        print(f"Processing bootstrap {b+1}/{bootstraps}")
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
