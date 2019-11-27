import pytest
from sklearn.datasets.samples_generator import make_circles

from src.dbscan import DBSCAN


def _are_lists_isomorphic(list_1, list_2):
    if len(list_1) != len(list_2):
        return False

    distinct_elements_1 = set(list_1)
    distinct_elements_2 = set(list_2)

    if len(distinct_elements_1) != len(distinct_elements_2):
        return False

    mappings = [(item_1, item_2) for (item_1, item_2)
                in zip(list_1, list_2)]
    distinct_mappings = set(mappings)

    return len(distinct_elements_1) == len(distinct_mappings)


@pytest.fixture
def circle_data():
    X, true_labels = \
        make_circles(n_samples=100, noise=0.02, factor=0.5, random_state=123)

    return X, true_labels


def test_dbscan_finds_correct_cluster_labels(circle_data):
    X, true_labels = circle_data

    clustering = DBSCAN(eps=0.3, min_samples=5)
    clustering.fit(X)

    assert _are_lists_isomorphic(true_labels, clustering.labels)


def test_dbscan_marks_singleton_point_as_noise(circle_data):
    X, _ = circle_data
    X[-1] = [0, 0]

    clustering = DBSCAN(eps=0.3, min_samples=5)
    clustering.fit(X)

    assert clustering.labels[-1] == DBSCAN.CLUSTER_LABEL_NOISE
