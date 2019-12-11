import numpy as np
import pytest
from sklearn.datasets.samples_generator import make_blobs

from src.incrementaldbscan.incrementaldbscan import IncrementalDBSCAN


@pytest.fixture
def incdbscan():
    return IncrementalDBSCAN(eps=1.5, min_pts=4)


@pytest.fixture
def blob_in_middle():
    values, _ = make_blobs(
        n_samples=10,
        centers=[[0, 0]],
        n_features=2,
        cluster_std=0.5,
        random_state=123
    )
    ids = np.arange(len(values))
    return values, ids


@pytest.fixture
def object_far_away():
    value = np.array([10, 10])
    id_ = 'FARAWAY'
    return value, id_


def test_new_single_object_is_labeled_as_noise(incdbscan, object_far_away):
    object_value, object_id = object_far_away
    incdbscan.add_object(object_value, object_id)

    assert incdbscan.labels[object_id] == incdbscan.CLUSTER_LABEL_NOISE


def test_new_object_far_from_cluster_is_labeled_as_noise(
        incdbscan,
        blob_in_middle,
        object_far_away):

    blob_values, blob_ids = blob_in_middle
    object_value, object_id = object_far_away

    incdbscan.add_objects(blob_values, blob_ids)
    incdbscan.add_object(object_value, object_id)

    assert incdbscan.labels[object_id] == incdbscan.CLUSTER_LABEL_NOISE


def test_new_border_object_gets_label_from_core(incdbscan):
    cluster = np.array([
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
    ])
    ids_in_cluster = list(range(len(cluster)))

    new_border_object_value = np.array([1 + incdbscan.eps, 1])
    new_border_object_id = max(ids_in_cluster) + 1

    incdbscan.add_objects(cluster, ids_in_cluster)
    incdbscan.add_object(new_border_object_value, new_border_object_id)

    assert incdbscan.labels[new_border_object_id] == \
        incdbscan.labels[ids_in_cluster[-1]]


def test_labels_are_noise_until_not_enough_objects_in_cluster(
        incdbscan,
        blob_in_middle):

    blob_values, blob_ids = blob_in_middle

    for i, (object_value, object_id) in enumerate(zip(blob_values, blob_ids)):
        incdbscan.add_object(object_value, object_id)

        expected_label = (
            incdbscan.CLUSTER_LABEL_NOISE if i + 1 < incdbscan.min_pts
            else incdbscan.CLUSTER_LABEL_FIRST_CLUSTER
        )

        for j in range(i + 1):
            assert incdbscan.labels[blob_ids[j]] == expected_label


def test_new_clusters_get_new_labels(incdbscan, blob_in_middle):
    distance = 10
    cluster_1_values, cluster_1_ids = blob_in_middle
    cluster_1_expected_label = incdbscan.CLUSTER_LABEL_FIRST_CLUSTER

    incdbscan.add_objects(cluster_1_values, cluster_1_ids)
    for object_id in cluster_1_ids:
        assert incdbscan.labels[object_id] == cluster_1_expected_label

    cluster_2_values = cluster_1_values + distance * 1
    cluster_2_ids = cluster_1_ids + len(cluster_1_ids) * 1
    cluster_2_expected_label = incdbscan.CLUSTER_LABEL_FIRST_CLUSTER + 1

    incdbscan.add_objects(cluster_2_values, cluster_2_ids)
    for object_id in cluster_2_ids:
        assert incdbscan.labels[object_id] == cluster_2_expected_label

    cluster_3_values = cluster_1_values + distance * 2
    cluster_3_ids = cluster_1_ids + len(cluster_1_ids) * 2
    cluster_3_expected_label = incdbscan.CLUSTER_LABEL_FIRST_CLUSTER + 2

    incdbscan.add_objects(cluster_3_values, cluster_3_ids)
    for object_id in cluster_3_ids:
        assert incdbscan.labels[object_id] == cluster_3_expected_label
