import numpy as np
from pytest import approx

from src.distance import euclidean_distance, get_neighbors


def test_euclidean_distance():
    a = np.array([0, 3, 4, 5])
    b = np.array([7, 6, 3, -1])

    expected_distance = 9.747
    assert euclidean_distance(a, b) == approx(expected_distance, abs=1e-3)


def test_get_neighbors_returns_indices_only_of_close_items():
    x = np.array([4, 4])
    close_item_1 = np.array([1, 4])
    close_item_2 = np.array([4, 2])
    far_item = np.array([10, 11])

    index_close_item_1 = 0
    index_close_item_2 = 1
    index_far_item = 2

    items = [None] * 3
    items[index_close_item_1] = close_item_1
    items[index_close_item_2] = close_item_2
    items[index_far_item] = far_item

    neighbors = \
        get_neighbors(x, items, radius=3, distance_function=euclidean_distance)

    assert neighbors == [index_close_item_1, index_close_item_2]
