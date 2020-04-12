from typing import Iterable
from src.incrementaldbscan._objects import ObjectId

CLUSTER_LABEL_NOISE = -1
CLUSTER_LABEL_FIRST_CLUSTER = 0


def assert_cluster_label_of_ids(
        object_ids: Iterable[ObjectId],
        incdbscan_fit,
        label):

    for object_id in object_ids:
        assert incdbscan_fit.labels[object_id] == label


def add_objects_to_clustering_and_assert_membership(
        incdbscan,
        values: Iterable,
        ids_to_add: Iterable[ObjectId],
        expected_label):

    incdbscan.add_objects(values, ids_to_add)
    assert_cluster_label_of_ids(ids_to_add, incdbscan, expected_label)


def assert_split_creates_new_labels(
        incdbscan_fit,
        components: Iterable[Iterable[ObjectId]],
        previous_common_label):

    all_labels = set()

    for component in components:
        labels_within_component = set()

        for object_id in component:
            labels_within_component.add(incdbscan_fit.labels[object_id])

        assert len(labels_within_component) == 1
        all_labels.update(labels_within_component)

    assert previous_common_label in all_labels
    assert len(all_labels) == len(components)
    assert CLUSTER_LABEL_NOISE not in all_labels
