import numpy as np
import pytest
from sklearn.datasets.samples_generator import make_blobs

from src.incrementaldbscan import IncrementalDBSCAN


@pytest.fixture
def incdbscan():
    return IncrementalDBSCAN(eps=0.5, min_pts=5)


@pytest.fixture
def blob_in_middle():
    values, _ = make_blobs(
        n_samples=10,
        centers=[[0, 0]],
        n_features=2,
        cluster_std=0.5,
        random_state=123
    )
    ids = range(len(values))
    return values, ids


@pytest.fixture
def object_far_away():
    value = np.array([10, 10])
    id_ = 'FARAWAY'
    return value, id_


def test_new_single_point_is_labeled_as_noise(incdbscan, object_far_away):
    object_value, object_id = object_far_away
    incdbscan.add_object(object_value, object_id)

    assert incdbscan.labels[object_id] == incdbscan.CLUSTER_LABEL_NOISE


def test_new_point_far_away_is_labeled_as_noise(
        incdbscan,
        blob_in_middle,
        object_far_away):

    blob_values, blob_ids = blob_in_middle
    object_value, object_id = object_far_away

    incdbscan.add_objects(blob_values, blob_ids)
    incdbscan.add_object(object_value, object_id)

    assert incdbscan.labels[object_id] == incdbscan.CLUSTER_LABEL_NOISE
