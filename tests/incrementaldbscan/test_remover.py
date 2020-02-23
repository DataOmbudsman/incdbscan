import numpy as np

from tests.incrementaldbscan.conftest import EPS
from tests.incrementaldbscan.utils import (
    add_values_to_clustering_and_assert,
    assert_cluster_label_of_ids
)

CLUSTER_LABEL_NOISE = -1
CLUSTER_LABEL_FIRST_CLUSTER = 0


def test_after_removing_enough_objects_only_noise_remain(
        incdbscan4,
        blob_in_middle):

    blob_values, blob_ids = blob_in_middle
    incdbscan4.add_objects(blob_values, blob_ids)

    blob_ids = list(blob_ids)
    while blob_ids:
        object_id_to_remove = blob_ids.pop(-1)
        incdbscan4.remove_object(object_id_to_remove)

        expected_label = (
            CLUSTER_LABEL_NOISE
            if len(blob_ids) < incdbscan4.min_pts
            else CLUSTER_LABEL_FIRST_CLUSTER
        )

        assert_cluster_label_of_ids(blob_ids, incdbscan4, expected_label)


def test_removing_only_core_makes_borders_noise(incdbscan4, point_at_origin):
    core_value, core_id = point_at_origin
    incdbscan4.add_object(core_value, core_id)

    border_values = np.array([
        [EPS, 0],
        [0, EPS],
        [0, -EPS],
    ])
    border_ids = np.array([0, 1, 2])

    incdbscan4.add_objects(border_values, border_ids)

    incdbscan4.remove_object(core_id)

    assert_cluster_label_of_ids(border_ids, incdbscan4, CLUSTER_LABEL_NOISE)


def test_objects_losing_core_property_can_keep_cluster_id(
        incdbscan3,
        point_at_origin):

    point_to_remove_value, point_to_remove_id = point_at_origin

    core_values = np.array([
        [EPS, 0],
        [0, EPS],
        [EPS, EPS],
    ])
    core_ids = [0, 1, 2]

    all_values = np.vstack([point_to_remove_value, core_values])
    all_ids = [point_to_remove_id] + core_ids

    add_values_to_clustering_and_assert(
        incdbscan3, all_values, all_ids, CLUSTER_LABEL_FIRST_CLUSTER
    )

    incdbscan3.remove_object(point_to_remove_id)
    assert_cluster_label_of_ids(
        core_ids, incdbscan3, CLUSTER_LABEL_FIRST_CLUSTER)


def test_border_object_can_switch_to_other_cluster(
        incdbscan4,
        point_at_origin):

    border_value, border_id = point_at_origin
    incdbscan4.add_object(border_value, border_id)

    cluster_1_values = np.array([
        [EPS, 0],
        [EPS, EPS],
        [EPS, -EPS],
    ])
    cluster_1_ids = np.array([0, 1, 2])
    cluster_1_label = CLUSTER_LABEL_FIRST_CLUSTER

    cluster_2_values = np.array([
        [-EPS, 0],
        [-EPS, EPS],
        [-EPS, -EPS],
    ])
    cluster_2_ids = np.array([3, 4, 5])
    cluster_2_label = cluster_1_label + 1

    add_values_to_clustering_and_assert(
        incdbscan4,
        cluster_1_values,
        cluster_1_ids,
        cluster_1_label
    )

    add_values_to_clustering_and_assert(
        incdbscan4,
        cluster_2_values,
        cluster_2_ids,
        cluster_2_label
    )

    assert_cluster_label_of_ids([border_id], incdbscan4, cluster_2_label)

    incdbscan4.remove_object(cluster_2_ids[0])

    assert_cluster_label_of_ids([border_id], incdbscan4, cluster_1_label)


def test_borders_around_point_losing_core_property_can_become_noise(
        incdbscan4,
        point_at_origin):

    point_to_remove_value, point_to_remove_id = point_at_origin

    core_value = np.array([0, EPS])
    core_id = 1

    border_values = np.array([
        [0, EPS * 2],
        [EPS, EPS]
    ])
    border_ids = [2, 3]

    all_values = np.vstack([point_to_remove_value, core_value, border_values])
    all_ids_but_object_to_remove = border_ids + [core_id]
    all_ids = all_ids_but_object_to_remove + [point_to_remove_id]

    add_values_to_clustering_and_assert(
        incdbscan4, all_values, all_ids, CLUSTER_LABEL_FIRST_CLUSTER
    )

    incdbscan4.remove_object(point_to_remove_id)

    assert_cluster_label_of_ids(
        all_ids_but_object_to_remove,
        incdbscan4,
        CLUSTER_LABEL_NOISE
    )


def test_core_property_of_singleton_update_seed_is_kept_after_removal(
        incdbscan3,
        point_at_origin):

    point_to_remove_value, point_to_remove_id = point_at_origin

    core_values = np.array([
        [EPS, 0],
        [2 * EPS, 0],
        [2 * EPS, 0],
    ])
    core_ids = [1, 2, 3]

    lonely_value = np.array([-EPS, 0])
    lonely_id = 4

    all_values = np.vstack([point_to_remove_value, core_values, lonely_value])
    all_ids = [point_to_remove_id] + core_ids + [lonely_id]

    add_values_to_clustering_and_assert(
        incdbscan3, all_values, all_ids, CLUSTER_LABEL_FIRST_CLUSTER
    )

    incdbscan3.remove_object(point_to_remove_id)

    assert_cluster_label_of_ids(
        core_ids, incdbscan3, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_label_of_ids([lonely_id], incdbscan3, CLUSTER_LABEL_NOISE)


def test_cluster_id_of_single_component_update_seeds_is_kept_after_removal(
        incdbscan3,
        point_at_origin):

    point_to_remove_value, point_to_remove_id = point_at_origin

    core_values = np.array([
        [EPS, 0],
        [EPS, 0],
        [2 * EPS, 0],
    ])
    core_ids = [1, 2, 3]

    lonely_value = np.array([-EPS, 0])
    lonely_id = 4

    all_values = np.vstack([point_to_remove_value, core_values, lonely_value])
    all_ids = [point_to_remove_id] + core_ids + [lonely_id]

    add_values_to_clustering_and_assert(
        incdbscan3, all_values, all_ids, CLUSTER_LABEL_FIRST_CLUSTER
    )

    incdbscan3.remove_object(point_to_remove_id)

    assert_cluster_label_of_ids(
        core_ids, incdbscan3, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_label_of_ids([lonely_id], incdbscan3, CLUSTER_LABEL_NOISE)


def test_cluster_id_of_single_component_objects_is_kept_after_removal(
        incdbscan3,
        point_at_origin):

    point_to_remove_value, point_to_remove_id = point_at_origin

    core_values = np.array([
        [EPS, 0],
        [0, EPS],
        [EPS, EPS],
        [EPS, EPS],
    ])
    core_ids = [0, 1, 2, 3]

    all_values = np.vstack([point_to_remove_value, core_values])
    all_ids = [point_to_remove_id] + core_ids

    add_values_to_clustering_and_assert(
        incdbscan3, all_values, all_ids, CLUSTER_LABEL_FIRST_CLUSTER
    )

    incdbscan3.remove_object(point_to_remove_id)
    assert_cluster_label_of_ids(
        core_ids, incdbscan3, CLUSTER_LABEL_FIRST_CLUSTER)
