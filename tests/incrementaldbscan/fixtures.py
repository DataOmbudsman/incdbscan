from typing import Iterable

import numpy as np
import pytest
from sklearn.datasets.samples_generator import make_blobs

from src.incrementaldbscan.incrementaldbscan import IncrementalDBSCAN

EPS = 1.5


@pytest.fixture
def incdbscan3():
    return IncrementalDBSCAN(eps=EPS, min_pts=3)


@pytest.fixture
def incdbscan4():
    return IncrementalDBSCAN(eps=EPS, min_pts=4)


@pytest.fixture
def blob_in_middle():
    values, _ = make_blobs(
        n_samples=10,
        centers=[[0, 0]],
        n_features=2,
        cluster_std=0.4,
        random_state=123
    )
    ids = np.arange(len(values))
    return values, ids


@pytest.fixture
def object_far_away():
    value = np.array([10, 10])
    id_ = 'FARAWAY'
    return value, id_


@pytest.fixture
def point_at_origin():
    value = np.array([0, 0])
    id_ = 'NEW'
    return value, id_


def assert_cluster_label_of_ids(object_ids: Iterable, incdbscan_fit, label):
    for object_id in object_ids:
        assert incdbscan_fit.labels[object_id] == label


def add_values_to_clustering_and_assert(
        incdbscan,
        values: Iterable,
        ids: Iterable,
        expected_label):

    incdbscan.add_objects(values, ids)
    assert_cluster_label_of_ids(ids, incdbscan, expected_label)
