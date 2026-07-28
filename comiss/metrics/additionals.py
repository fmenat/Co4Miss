import numpy as np

def corr_between(A, B):
    """calculate correlation between two arrays

    Parameters
    ----------
    A : numpy array (n-samples x n-dims)
    B : numpy array (n-samples x n-dims)

    Returns
    -------
    correlation matrix between all features in both arrays
    """
    a_len = A.shape[1]
    return np.corrcoef(A, B, rowvar=False)[:a_len,a_len:]

def PearsonCorrelation(abs_magnitude=True):
    def metric(V1,V2):
        value = corr_between(V1,V2)
        return np.abs(value) if abs_magnitude else value
    return metric