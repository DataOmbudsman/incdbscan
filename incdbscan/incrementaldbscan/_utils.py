import joblib
from sklearn.utils.validation import check_array


def hash_(array):
    return joblib.hash(array)


def input_check(X):
    return check_array(X, dtype=float, accept_large_sparse=False)
