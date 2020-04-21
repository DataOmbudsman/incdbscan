from typing import Iterable

import numpy as np

CLUSTER_LABEL_NOISE = -1
CLUSTER_LABEL_FIRST_CLUSTER = 0


def assert_cluster_labels(incdbscan_fit, objects: Iterable, label):
    assert np.all(
        incdbscan_fit.get_cluster_labels(objects) == label
    )


def assert_two_objects_are_in_same_cluster(incdbscan_fit, object1, object2):
    assert incdbscan_fit.get_cluster_labels(object1) == \
        incdbscan_fit.get_cluster_labels(object2)


def assert_label_of_object_is_among_possible_ones(
        incdbscan_fit,
        obj,
        possible_labels):

    assert incdbscan_fit.get_cluster_labels(obj)[0] in possible_labels


def insert_objects_then_assert_cluster_labels(
        incdbscan,
        values: Iterable,
        expected_label):

    incdbscan.insert_objects(values)
    assert_cluster_labels(incdbscan, values, expected_label)


def assert_split_creates_new_labels_for_new_clusters(
        incdbscan_fit,
        clusters: Iterable[Iterable],
        previous_common_label):

    all_labels = set()

    for cluster in clusters:
        labels_within_cluster = set()

        for obj in cluster:
            labels_within_cluster.add(incdbscan_fit.get_cluster_labels(obj)[0])

        assert len(labels_within_cluster) == 1
        all_labels.update(labels_within_cluster)

    assert previous_common_label in all_labels
    assert len(all_labels) == len(clusters)
    assert CLUSTER_LABEL_NOISE not in all_labels


def reflect_horizontally(points):
    new_points = np.copy(points)
    new_points[:, 0] = np.negative(new_points[:, 0])
    return new_points
