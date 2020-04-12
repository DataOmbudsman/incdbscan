import numpy as np

from tests.incrementaldbscan.conftest import EPS
from tests.incrementaldbscan.utils import (
    add_objects_to_clustering_and_assert,
    assert_cluster_label_of_ids,
    CLUSTER_LABEL_NOISE,
    CLUSTER_LABEL_FIRST_CLUSTER
)


def test_new_single_object_is_labeled_as_noise(incdbscan4, object_far_away):
    object_value, object_id = object_far_away
    incdbscan4.add_object(object_value, object_id)

    assert incdbscan4.labels[object_id] == CLUSTER_LABEL_NOISE


def test_new_object_far_from_cluster_is_labeled_as_noise(
        incdbscan4,
        blob_in_middle,
        object_far_away):

    blob_values, blob_ids = blob_in_middle
    object_value, object_id = object_far_away

    incdbscan4.add_objects(blob_values, blob_ids)
    incdbscan4.add_object(object_value, object_id)

    assert incdbscan4.labels[object_id] == CLUSTER_LABEL_NOISE


def test_new_border_object_gets_label_from_core(incdbscan4):
    cluster = np.array([
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
    ])
    ids_in_cluster = list(range(len(cluster)))

    new_border_object_value = np.array([1 + EPS, 1])
    new_border_object_id = max(ids_in_cluster) + 1

    incdbscan4.add_objects(cluster, ids_in_cluster)
    incdbscan4.add_object(new_border_object_value, new_border_object_id)

    assert incdbscan4.labels[new_border_object_id] == \
        incdbscan4.labels[ids_in_cluster[-1]]


def test_labels_are_noise_only_until_not_enough_objects_in_cluster(
        incdbscan4,
        blob_in_middle):

    blob_values, blob_ids = blob_in_middle

    for i, (object_value, object_id) in enumerate(zip(blob_values, blob_ids)):
        incdbscan4.add_object(object_value, object_id)

        expected_label = (
            CLUSTER_LABEL_NOISE if i + 1 < incdbscan4.min_pts
            else CLUSTER_LABEL_FIRST_CLUSTER
        )

        assert_cluster_label_of_ids(blob_ids[:i+1], incdbscan4, expected_label)


def test_more_than_two_clusters_can_be_created(incdbscan4, blob_in_middle):

    cluster_1_values, cluster_1_ids = blob_in_middle
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    add_objects_to_clustering_and_assert(
        incdbscan4, cluster_1_values, cluster_1_ids, cluster_1_expected_label)

    cluster_2_values, cluster_2_ids = \
        cluster_1_values + 10, cluster_1_ids + 10
    cluster_2_expected_label = cluster_1_expected_label + 1

    add_objects_to_clustering_and_assert(
        incdbscan4, cluster_2_values, cluster_2_ids, cluster_2_expected_label)

    cluster_3_values, cluster_3_ids = \
        cluster_2_values + 10, cluster_2_ids + 10
    cluster_3_expected_label = cluster_2_expected_label + 1

    add_objects_to_clustering_and_assert(
        incdbscan4, cluster_3_values, cluster_3_ids, cluster_3_expected_label)


def test_two_clusters_can_be_born_at_the_same_time(
        incdbscan4,
        point_at_origin):

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

    incdbscan4.add_objects(cluster_1_values, cluster_1_ids)
    incdbscan4.add_objects(cluster_2_values, cluster_2_ids)

    assert_cluster_label_of_ids(cluster_1_ids, incdbscan4, CLUSTER_LABEL_NOISE)
    assert_cluster_label_of_ids(cluster_2_ids, incdbscan4, CLUSTER_LABEL_NOISE)

    new_object_value, new_object_id = point_at_origin
    incdbscan4.add_object(new_object_value, new_object_id)

    cluster_1_label_expected = incdbscan4.labels[cluster_1_ids[0]]
    assert_cluster_label_of_ids(
        cluster_1_ids, incdbscan4, cluster_1_label_expected)

    cluster_2_label_expected = \
        CLUSTER_LABEL_FIRST_CLUSTER + 1 - cluster_1_label_expected
    assert_cluster_label_of_ids(
        cluster_2_ids, incdbscan4, cluster_2_label_expected)

    assert incdbscan4.labels[new_object_id] in set((
        cluster_1_label_expected,
        cluster_2_label_expected
    ))


def test_absorption_with_noise(incdbscan3, point_at_origin):
    expected_cluster_label = CLUSTER_LABEL_FIRST_CLUSTER

    cluster_values = np.array([
        [EPS, 0],
        [EPS * 2, 0],
        [EPS * 3, 0],
    ])
    cluster_ids = [0, 1, 2]

    add_objects_to_clustering_and_assert(
        incdbscan3, cluster_values, cluster_ids, expected_cluster_label)

    noise_value, noise_id = np.array([0, EPS]), 'NOISE'

    add_objects_to_clustering_and_assert(
        incdbscan3, [noise_value], [noise_id], CLUSTER_LABEL_NOISE)

    new_object_value, new_object_id = point_at_origin

    add_objects_to_clustering_and_assert(
        incdbscan3,
        [new_object_value],
        [new_object_id],
        expected_cluster_label
    )

    assert_cluster_label_of_ids([noise_id], incdbscan3, expected_cluster_label)


def test_merge_two_clusters(incdbscan3, point_at_origin):
    cluster_1_values = np.array([
        [EPS, 0],
        [EPS * 2, 0],
        [EPS * 3, 0],
        [EPS * 4, 0],
    ])
    cluster_1_ids = [0, 1, 2, 3]
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    add_objects_to_clustering_and_assert(
        incdbscan3, cluster_1_values, cluster_1_ids, cluster_1_expected_label
    )

    cluster_2_values = np.array([
        [-EPS, 0],
        [-EPS * 2, 0],
        [-EPS * 3, 0],
        [-EPS * 4, 0],
    ])
    cluster_2_ids = [4, 5, 6, 7]
    cluster_2_expected_label = cluster_1_expected_label + 1

    add_objects_to_clustering_and_assert(
        incdbscan3, cluster_2_values, cluster_2_ids, cluster_2_expected_label
    )

    new_object_value, new_object_id = point_at_origin
    merged_cluster_expected_label = \
        max([cluster_1_expected_label, cluster_2_expected_label])

    add_objects_to_clustering_and_assert(
        incdbscan3,
        [new_object_value],
        [new_object_id],
        merged_cluster_expected_label
    )

    assert_cluster_label_of_ids(
        cluster_1_ids, incdbscan3, merged_cluster_expected_label)
    assert_cluster_label_of_ids(
        cluster_2_ids, incdbscan3, merged_cluster_expected_label)


def test_merger_and_creation_can_happen_at_the_same_time(
        incdbscan4,
        point_at_origin):

    cluster_1_values = np.array([
        [EPS, EPS],
        [EPS, EPS * 2],
        [EPS, EPS * 2],
    ])
    cluster_1_ids = [0, 1, 2]
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    cluster_2_values = np.array([
        [EPS, -EPS],
        [EPS, -EPS * 2],
        [EPS, -EPS * 2],
    ])
    cluster_2_ids = [3, 4, 5]
    cluster_2_expected_label = cluster_1_expected_label + 1

    bridge_point_value, bridge_point_id = np.array([EPS, 0]), 'bridge'

    incdbscan4.add_objects(cluster_1_values, cluster_1_ids)
    incdbscan4.add_object(bridge_point_value, bridge_point_id)
    incdbscan4.add_objects(cluster_2_values, cluster_2_ids)

    assert_cluster_label_of_ids(
        cluster_1_ids, incdbscan4, cluster_1_expected_label)
    assert_cluster_label_of_ids(
        cluster_2_ids, incdbscan4, cluster_2_expected_label)
    assert incdbscan4.labels[bridge_point_id]\
        in {cluster_2_expected_label, cluster_2_expected_label}

    merged_cluster_expected_label = incdbscan4.labels[bridge_point_id]

    pre_cluster_values = np.array([
        [-EPS, 0],
        [-EPS * 2, 0],
        [-EPS * 2, 0],
    ])
    pre_cluster_ids = [6, 7, 8]
    cluster_3_expected_label = cluster_2_expected_label + 1

    add_objects_to_clustering_and_assert(
        incdbscan4,
        pre_cluster_values,
        pre_cluster_ids,
        CLUSTER_LABEL_NOISE
    )

    new_object_value, new_object_id = point_at_origin
    incdbscan4.add_object(new_object_value, new_object_id)

    assert_cluster_label_of_ids(
        cluster_1_ids, incdbscan4, merged_cluster_expected_label)
    assert_cluster_label_of_ids(
        cluster_2_ids, incdbscan4, merged_cluster_expected_label)
    assert_cluster_label_of_ids(
        [bridge_point_id], incdbscan4, merged_cluster_expected_label)
    assert_cluster_label_of_ids(
        pre_cluster_ids, incdbscan4, cluster_3_expected_label)
    assert incdbscan4.labels[new_object_id]\
        in {merged_cluster_expected_label, cluster_3_expected_label}
