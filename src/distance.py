import numpy as np


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def get_neighbors(x, items, radius, distance_function=euclidean_distance):
    """Naive linear search. Returns indices of neighbors"""
    return [ix for ix, item in enumerate(items)
            if distance_function(x, item) <= radius]
