import numpy as np
from conftest import EPS

from testutils import (
    CLUSTER_LABEL_FIRST_CLUSTER,
    CLUSTER_LABEL_NOISE,
    assert_cluster_labels,
    assert_label_of_object_is_among_possible_ones,
    assert_split_creates_new_labels_for_new_clusters,
    insert_objects_then_assert_cluster_labels,
    reflect_horizontally
)


def test_after_deleting_enough_objects_only_noise_remain(
        incdbscan4,
        blob_in_middle):

    incdbscan4.insert(blob_in_middle)

    for i in range(len(blob_in_middle) - 1):
        object_to_delete = blob_in_middle[[i]]

        incdbscan4.delete(object_to_delete)

        expected_label = (
            CLUSTER_LABEL_NOISE
            if i > incdbscan4.min_pts + 1
            else CLUSTER_LABEL_FIRST_CLUSTER
        )

        assert_cluster_labels(incdbscan4, blob_in_middle[i+1:], expected_label)


def test_deleting_cores_only_makes_borders_noise(incdbscan4, point_at_origin):
    point_to_delete = point_at_origin
    incdbscan4.insert(point_to_delete)

    border = np.array([
        [EPS, 0],
        [0, EPS],
        [0, -EPS],
    ])

    incdbscan4.insert(border)
    incdbscan4.delete(point_to_delete)

    assert_cluster_labels(incdbscan4, border, CLUSTER_LABEL_NOISE)


def test_objects_losing_core_property_can_keep_cluster_id(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    core_points = np.array([
        [EPS, 0],
        [0, EPS],
        [EPS, EPS],
    ])

    all_points = np.vstack([point_to_delete, core_points])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)
    assert_cluster_labels(incdbscan3, core_points, CLUSTER_LABEL_FIRST_CLUSTER)


def test_border_object_can_switch_to_other_cluster(
        incdbscan4,
        point_at_origin):

    border = point_at_origin
    incdbscan4.insert(border)

    cluster_1 = np.array([
        [EPS, 0],
        [EPS, EPS],
        [EPS, -EPS],
    ])
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    cluster_2 = reflect_horizontally(cluster_1)
    cluster_2_expected_label = cluster_1_expected_label + 1

    insert_objects_then_assert_cluster_labels(
        incdbscan4, cluster_1, cluster_1_expected_label)

    insert_objects_then_assert_cluster_labels(
        incdbscan4, cluster_2, cluster_2_expected_label
    )

    assert_cluster_labels(incdbscan4, border, cluster_2_expected_label)

    incdbscan4.delete(cluster_2[[0]])

    assert_cluster_labels(incdbscan4, border, cluster_1_expected_label)


def test_borders_around_point_losing_core_property_can_become_noise(
        incdbscan4,
        point_at_origin):

    point_to_delete = point_at_origin

    core = np.array([[0, EPS]])

    border = np.array([
        [0, EPS * 2],
        [EPS, EPS]
    ])

    all_points = np.vstack([point_to_delete, core, border])
    all_points_but_point_to_delete = np.vstack([core, border])

    insert_objects_then_assert_cluster_labels(
        incdbscan4, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan4.delete(point_to_delete)

    assert_cluster_labels(
        incdbscan4, all_points_but_point_to_delete, CLUSTER_LABEL_NOISE)


def test_core_property_of_singleton_update_seed_is_kept_after_deletion(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    cores = np.array([
        [EPS, 0],
        [2 * EPS, 0],
        [2 * EPS, 0],
    ])

    lonely = np.array([[-EPS, 0]])

    all_points = np.vstack([point_to_delete, cores, lonely])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_cluster_labels(incdbscan3, cores, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(incdbscan3, lonely, CLUSTER_LABEL_NOISE)


def test_cluster_id_of_single_component_update_seeds_is_kept_after_deletion(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    cores = np.array([
        [EPS, 0],
        [EPS, 0],
        [2 * EPS, 0],
    ])

    lonely = np.array([[-EPS, 0]])

    all_points = np.vstack([point_to_delete, cores, lonely])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_cluster_labels(incdbscan3, cores, CLUSTER_LABEL_FIRST_CLUSTER)
    assert_cluster_labels(incdbscan3, lonely, CLUSTER_LABEL_NOISE)


def test_cluster_id_of_single_component_objects_is_kept_after_deletion(
        incdbscan3,
        point_at_origin):

    point_to_delete = point_at_origin

    cores = np.array([
        [EPS, 0],
        [0, EPS],
        [EPS, EPS],
        [EPS, EPS],
    ])

    all_points = np.vstack([point_to_delete, cores])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_cluster_labels(incdbscan3, cores, CLUSTER_LABEL_FIRST_CLUSTER)


def test_simple_two_way_split(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left
    points_right = reflect_horizontally(points_left)

    all_points = np.vstack([point_to_delete, points_left, points_right])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3, [points_left, points_right], CLUSTER_LABEL_FIRST_CLUSTER)


def test_simple_two_way_split_with_noise(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left,
        three_points_on_the_top,
        three_points_at_the_bottom):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left
    points_top = three_points_on_the_top
    points_bottom = three_points_at_the_bottom[:-1]

    all_points = np.vstack([
        point_to_delete,
        points_left,
        points_top,
        points_bottom
    ])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3, [points_left, points_top], CLUSTER_LABEL_FIRST_CLUSTER)

    assert_cluster_labels(incdbscan3, points_bottom, CLUSTER_LABEL_NOISE)


def test_three_way_split(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left,
        three_points_on_the_top,
        three_points_at_the_bottom):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left
    points_top = three_points_on_the_top
    points_bottom = three_points_at_the_bottom

    all_points = np.vstack([
        point_to_delete,
        points_left,
        points_top,
        points_bottom
    ])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3,
        [points_left, points_top, points_bottom],
        CLUSTER_LABEL_FIRST_CLUSTER
    )


def test_simultaneous_split_and_non_split(
        incdbscan3,
        point_at_origin,
        three_points_on_the_left):

    point_to_delete = point_at_origin
    points_left = three_points_on_the_left

    points_right = np.array([
        [0, EPS],
        [0, -EPS],
        [EPS, 0],
        [EPS, EPS],
        [EPS, -EPS],
    ])

    all_points = np.vstack([point_to_delete, points_left, points_right])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan3.delete(point_to_delete)

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan3, [points_left, points_right], CLUSTER_LABEL_FIRST_CLUSTER)


def test_two_way_split_with_non_dense_bridge(incdbscan4, point_at_origin):
    point_to_delete = bridge_point = point_at_origin

    points_left = np.array([
        [0, -EPS],
        [0, -EPS * 2],
        [0, -EPS * 2],
        [0, -EPS * 3],
        [0, -EPS * 3],
    ])

    points_right = np.array([
        [0, EPS],
        [0, EPS * 2],
        [0, EPS * 2],
        [0, EPS * 3],
        [0, EPS * 3],
    ])

    all_points = np.vstack([
        bridge_point, point_to_delete, points_left, points_right
    ])

    insert_objects_then_assert_cluster_labels(
        incdbscan4, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan4.delete(point_to_delete)

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan4, [points_left, points_right], CLUSTER_LABEL_FIRST_CLUSTER)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point,
        {CLUSTER_LABEL_FIRST_CLUSTER, CLUSTER_LABEL_FIRST_CLUSTER + 1}
    )


def test_simultaneous_splits_within_two_clusters(
        incdbscan4,
        point_at_origin,
        hourglass_on_the_right):

    point_to_delete = point_at_origin
    points_right = hourglass_on_the_right
    points_left = reflect_horizontally(points_right)

    incdbscan4.insert(point_to_delete)

    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER
    insert_objects_then_assert_cluster_labels(
        incdbscan4, points_left, cluster_1_expected_label)

    cluster_2_expected_label = CLUSTER_LABEL_FIRST_CLUSTER + 1
    insert_objects_then_assert_cluster_labels(
        incdbscan4, points_right, cluster_2_expected_label)

    incdbscan4.delete(point_to_delete)

    expected_clusters = [
        points_left[:3], points_left[-3:], points_right[:3], points_right[-3:]
    ]

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan4, expected_clusters, CLUSTER_LABEL_FIRST_CLUSTER)

    expected_cluster_labels_left = {
        incdbscan4.get_cluster_labels(points_left[[2]])[0],
        incdbscan4.get_cluster_labels(points_left[[4]])[0],
    }

    assert_label_of_object_is_among_possible_ones(
        incdbscan4, points_left[[3]], expected_cluster_labels_left)

    expected_cluster_labels_right = {
        incdbscan4.get_cluster_labels(points_right[[2]])[0],
        incdbscan4.get_cluster_labels(points_right[[4]])[0]
    }

    assert_label_of_object_is_among_possible_ones(
        incdbscan4, points_right[[3]], expected_cluster_labels_right)


def test_two_non_dense_bridges(incdbscan4, point_at_origin):
    point_to_delete = point_at_origin

    points_left = np.array([
        [-EPS, 0],
        [-EPS, 0],
        [-EPS, -EPS],
        [-EPS, -EPS],
        [-EPS, -EPS * 2],
    ])
    points_right = reflect_horizontally(points_left)

    points_top = np.array([
        [0, EPS],
        [0, EPS],
        [0, EPS * 2],
        [0, EPS * 2],
        [0, EPS * 3],
        [0, EPS * 3],
        [0, EPS * 4],
        [0, EPS * 4],
    ])

    bottom_bridge = np.array([[0, -EPS * 2]])

    all_points = np.vstack([
        point_to_delete, points_left, points_right, points_top, bottom_bridge
    ])

    insert_objects_then_assert_cluster_labels(
        incdbscan4, all_points, CLUSTER_LABEL_FIRST_CLUSTER)

    incdbscan4.delete(point_to_delete)

    expected_clusters = [points_left, points_right, points_top]

    assert_split_creates_new_labels_for_new_clusters(
        incdbscan4, expected_clusters, CLUSTER_LABEL_FIRST_CLUSTER)
