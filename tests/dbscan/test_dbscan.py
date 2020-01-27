import pytest
from sklearn.datasets import make_circles

from src.dbscan.dbscan import DBSCAN
from tests.dbscan.utils import are_lists_isomorphic


@pytest.fixture
def circle_data():
    X, true_labels = \
        make_circles(n_samples=100, noise=0.02, factor=0.5, random_state=123)

    return X, true_labels


def test_dbscan_finds_correct_cluster_labels(circle_data):
    X, true_labels = circle_data

    clustering = DBSCAN(eps=0.3, min_pts=5)
    clustering.fit(X)

    assert are_lists_isomorphic(true_labels, clustering.labels)


def test_dbscan_marks_singleton_point_as_noise(circle_data):
    X, _ = circle_data
    X[-1] = [0, 0]

    clustering = DBSCAN(eps=0.3, min_pts=5)
    clustering.fit(X)

    assert clustering.labels[-1] == DBSCAN.CLUSTER_LABEL_NOISE
