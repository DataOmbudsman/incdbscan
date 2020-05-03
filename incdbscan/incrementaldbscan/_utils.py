from sklearn.utils.validation import check_array
import xxhash


def hash_(array):
    return xxhash.xxh32(array.tobytes()).hexdigest()


def input_check(X):
    return check_array(X, dtype=float, accept_large_sparse=False)
