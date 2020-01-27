from typing import Iterable


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
