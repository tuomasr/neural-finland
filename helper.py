def stand_data(x):
    """
    Standardize each column of x to have zero mean and unit variance
    """

    return (x-x.mean(axis=0))/x.std(axis=0)

def norm_data(x, lb=0, ub=1):
    """
    Normalize each column of x to the interval [lb, ub]
    """

    x = (x-x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) # normalize to range [0, 1]

    return (ub-lb)*x + lb # scale the data to [lb, ub]

def norm_data_reverse(x, lb, ub, x_min, x_max):
    """
    Scale normalized data in the interval [lb, ub] back to original values
    """
    return (x_max-x_min)/(ub-lb)*(x-lb) + x_min