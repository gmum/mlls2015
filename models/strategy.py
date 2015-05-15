import numpy as np


def random_query(X, batch_size, seed):
    np.random.seed(seed)
    return np.random.randint(0, X.shape[0], size=batch_size)
