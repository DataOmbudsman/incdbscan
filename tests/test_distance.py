import numpy as np
from pytest import approx

from src.distance import euclidean_distance, get_neighbors


def test_euclidean_distance():
    a = np.array([0, 3, 4, 5])
    b = np.array([7, 6, 3, -1])

    expected_distance = 9.747
    assert euclidean_distance(a, b) == approx(expected_distance, abs=1e-3)


def test_get_neighbors_returns_only_close_items():
    x = np.array([4, 4])
    close_item_1 = np.array([1, 4])
    close_item_2 = np.array([4, 2])
    far_item = np.array([10, 11])

    close_items = [close_item_1, close_item_2]
    far_items = [far_item]
    items = close_items + far_items

    neighbors = \
        get_neighbors(x, items, radius=3, distance_function=euclidean_distance)

    assert neighbors == close_items
