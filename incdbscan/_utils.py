import numpy as np
from sklearn.utils.validation import check_array

BYTE_LENGTH = 4
D_TYPE = float


def encode_(array):
    length_bytes = len(array).to_bytes(BYTE_LENGTH, byteorder="little")
    val = length_bytes + array.tobytes()
    return val


def decode_(encoded_bytes):
    length = int.from_bytes(encoded_bytes[:BYTE_LENGTH], byteorder="little")
    array = np.frombuffer(encoded_bytes[BYTE_LENGTH:], dtype=D_TYPE).reshape((length,))
    return array


def input_check(X):
    return check_array(X, dtype=D_TYPE, accept_large_sparse=False)
