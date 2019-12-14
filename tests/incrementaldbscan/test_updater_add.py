from typing import Iterable

import numpy as np
import pytest
from sklearn.datasets.samples_generator import make_blobs

from src.incrementaldbscan.incrementaldbscan import IncrementalDBSCAN

EPS = 1.5


@pytest.fixture
def incdbscan():
    return IncrementalDBSCAN(eps=EPS, min_pts=4)


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

    new_border_object_value = np.array([1 + EPS, 1])
    new_border_object_id = max(ids_in_cluster) + 1

    incdbscan.add_objects(cluster, ids_in_cluster)
    incdbscan.add_object(new_border_object_value, new_border_object_id)

    assert incdbscan._objects.objects[new_border_object_id].neighbor_count < \
        incdbscan.min_pts

    assert incdbscan.labels[new_border_object_id] == \
        incdbscan.labels[ids_in_cluster[-1]]


def test_labels_are_noise_only_until_not_enough_objects_in_cluster(
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


def test_more_than_two_clusters_can_be_created(incdbscan, blob_in_middle):

    cluster_1_values, cluster_1_ids = blob_in_middle
    cluster_1_expected_label = incdbscan.CLUSTER_LABEL_FIRST_CLUSTER

    add_values_to_clustering_and_assert(
        incdbscan, cluster_1_values, cluster_1_ids, cluster_1_expected_label)

    cluster_2_values, cluster_2_ids = \
        cluster_1_values + 10, cluster_1_ids + 10
    cluster_2_expected_label = cluster_1_expected_label + 1

    add_values_to_clustering_and_assert(
        incdbscan, cluster_2_values, cluster_2_ids, cluster_2_expected_label)

    cluster_3_values, cluster_3_ids = \
        cluster_2_values + 10, cluster_2_ids + 10
    cluster_3_expected_label = cluster_2_expected_label + 1

    add_values_to_clustering_and_assert(
        incdbscan, cluster_3_values, cluster_3_ids, cluster_3_expected_label)


def test_two_clusters_can_be_born_at_the_same_time(incdbscan):
    cluster_1_values = np.array([
        [EPS * 1, 0],
        [EPS * 2, 0],
        [EPS * 2, 0],
    ])
    cluster_1_ids = np.array([0, 1, 2])

    cluster_2_values = np.array([
        [-EPS * 1, 0],
        [-EPS * 2, 0],
        [-EPS * 2, 0],
    ])
    cluster_2_ids = np.array([3, 4, 5])

    incdbscan.add_objects(cluster_1_values, cluster_1_ids)
    incdbscan.add_objects(cluster_2_values, cluster_2_ids)

    assert_cluster_label_of_ids(
        cluster_1_ids, incdbscan, incdbscan.CLUSTER_LABEL_NOISE)

    assert_cluster_label_of_ids(
        cluster_2_ids, incdbscan, incdbscan.CLUSTER_LABEL_NOISE)

    new_object_value = np.array([0, 0])
    new_object_id = 6

    incdbscan.add_object(new_object_value, new_object_id)

    cluster_1_label_expected = incdbscan.labels[cluster_1_ids[0]]
    assert_cluster_label_of_ids(
        cluster_1_ids, incdbscan, cluster_1_label_expected)

    cluster_2_label_expected = \
        incdbscan.CLUSTER_LABEL_FIRST_CLUSTER + 1 - cluster_1_label_expected
    assert_cluster_label_of_ids(
        cluster_2_ids, incdbscan, cluster_2_label_expected)

    assert incdbscan.labels[new_object_id] in set((
        cluster_1_label_expected,
        cluster_2_label_expected
    ))


def test_absorption_with_noise():
    incdbscan = IncrementalDBSCAN(EPS, min_pts=3)
    expected_cluster_id = incdbscan.CLUSTER_LABEL_FIRST_CLUSTER

    cluster_values = np.array([
        [EPS, 0],
        [EPS * 2, 0],
        [EPS * 3, 0],
    ])
    cluster_ids = [0, 1, 2]

    add_values_to_clustering_and_assert(
        incdbscan, cluster_values, cluster_ids, expected_cluster_id)

    noise_value, noise_id = np.array([0, EPS]), 'NOISE'

    add_values_to_clustering_and_assert(
        incdbscan, [noise_value], [noise_id], incdbscan.CLUSTER_LABEL_NOISE)

    new_object_value, new_object_id = np.array([0, 0]), 'NEW'

    add_values_to_clustering_and_assert(
        incdbscan, [new_object_value], [new_object_id], expected_cluster_id)
    assert_cluster_label_of_ids([noise_id], incdbscan, expected_cluster_id)
