import numpy as np
from conftest import EPS

from testutils import (
    CLUSTER_LABEL_FIRST_CLUSTER,
    CLUSTER_LABEL_NOISE,
    assert_cluster_labels,
    assert_label_of_object_is_among_possible_ones,
    assert_two_objects_are_in_same_cluster,
    insert_objects_then_assert_cluster_labels,
    reflect_horizontally
)


def test_new_single_object_is_labeled_as_noise(incdbscan4, object_far_away):
    incdbscan4.insert(object_far_away)
    assert_cluster_labels(incdbscan4, object_far_away, CLUSTER_LABEL_NOISE)


def test_new_object_far_from_cluster_is_labeled_as_noise(
        incdbscan4,
        blob_in_middle,
        object_far_away):

    incdbscan4.insert(blob_in_middle)
    incdbscan4.insert(object_far_away)

    assert_cluster_labels(incdbscan4, object_far_away, CLUSTER_LABEL_NOISE)


def test_new_border_object_gets_label_from_core(incdbscan4):
    cluster = np.array([
        [1., 1.],
        [0., 1.],
        [1., 0.],
        [0., 0.],
    ])

    new_border_object = np.array([[1 + EPS, 1]])

    incdbscan4.insert(cluster)
    incdbscan4.insert(new_border_object)

    print(incdbscan4.get_cluster_labels(cluster[[0]]))
    print(incdbscan4.get_cluster_labels(new_border_object))

    assert_two_objects_are_in_same_cluster(
        incdbscan4, cluster[[0]], new_border_object)


def test_labels_are_noise_only_until_not_enough_objects_in_cluster(
        incdbscan4,
        blob_in_middle):

    for i in range(len(blob_in_middle)):
        incdbscan4.insert(blob_in_middle[[i]])

        expected_label = (
            CLUSTER_LABEL_NOISE if i + 1 < incdbscan4.min_pts
            else CLUSTER_LABEL_FIRST_CLUSTER
        )

        assert_cluster_labels(incdbscan4, blob_in_middle[:i+1], expected_label)


def test_more_than_two_clusters_can_be_created(incdbscan4, blob_in_middle):
    cluster_1 = blob_in_middle
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    insert_objects_then_assert_cluster_labels(
        incdbscan4, cluster_1, cluster_1_expected_label)

    cluster_2 = cluster_1 + 10
    cluster_2_expected_label = cluster_1_expected_label + 1

    insert_objects_then_assert_cluster_labels(
        incdbscan4, cluster_2, cluster_2_expected_label)

    cluster_3 = cluster_2 + 10
    cluster_3_expected_label = cluster_2_expected_label + 1

    insert_objects_then_assert_cluster_labels(
        incdbscan4, cluster_3, cluster_3_expected_label)


def test_two_clusters_can_be_born_at_the_same_time(
        incdbscan4,
        point_at_origin):

    cluster_1 = np.array([
        [EPS * 1, 0],
        [EPS * 2, 0],
        [EPS * 2, 0],
    ])

    cluster_2 = reflect_horizontally(cluster_1)

    incdbscan4.insert(cluster_1)
    incdbscan4.insert(cluster_2)

    assert_cluster_labels(incdbscan4, cluster_1, CLUSTER_LABEL_NOISE)
    assert_cluster_labels(incdbscan4, cluster_2, CLUSTER_LABEL_NOISE)

    new_object = point_at_origin
    incdbscan4.insert(new_object)

    cluster_1_label_expected = incdbscan4.get_cluster_labels(cluster_1[[0]])[0]
    assert_cluster_labels(incdbscan4, cluster_1, cluster_1_label_expected)

    cluster_2_label_expected = \
        CLUSTER_LABEL_FIRST_CLUSTER + 1 - cluster_1_label_expected
    assert_cluster_labels(incdbscan4, cluster_2, cluster_2_label_expected)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        new_object,
        {cluster_1_label_expected, cluster_2_label_expected}
    )


def test_absorption_with_noise(incdbscan3, point_at_origin):
    expected_cluster_label = CLUSTER_LABEL_FIRST_CLUSTER

    cluster_values = np.array([
        [EPS, 0],
        [EPS * 2, 0],
        [EPS * 3, 0],
    ])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, cluster_values, expected_cluster_label)

    noise = np.array([[0, EPS]])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, noise, CLUSTER_LABEL_NOISE)

    new_object_value = point_at_origin

    insert_objects_then_assert_cluster_labels(
        incdbscan3, new_object_value, expected_cluster_label)

    assert_cluster_labels(incdbscan3, noise, expected_cluster_label)


def test_merge_two_clusters(incdbscan3, point_at_origin):
    cluster_1 = np.array([
        [EPS, 0],
        [EPS * 2, 0],
        [EPS * 3, 0],
        [EPS * 4, 0],
    ])
    cluster_1_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    insert_objects_then_assert_cluster_labels(
        incdbscan3, cluster_1, cluster_1_expected_label)

    cluster_2 = reflect_horizontally(cluster_1)
    cluster_2_expected_label = cluster_1_expected_label + 1

    insert_objects_then_assert_cluster_labels(
        incdbscan3, cluster_2, cluster_2_expected_label)

    new_object = point_at_origin
    merged_cluster_expected_label = \
        max([cluster_1_expected_label, cluster_2_expected_label])

    insert_objects_then_assert_cluster_labels(
        incdbscan3, new_object, merged_cluster_expected_label)

    assert_cluster_labels(incdbscan3, cluster_1, merged_cluster_expected_label)
    assert_cluster_labels(incdbscan3, cluster_2, merged_cluster_expected_label)


def test_merger_and_creation_can_happen_at_the_same_time(
        incdbscan4,
        point_at_origin,
        hourglass_on_the_right):

    # Insert objects to the right
    hourglass = hourglass_on_the_right

    top_right = hourglass[:3]
    top_right_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    bottom_right = hourglass[-3:]
    bottom_right_expected_label = top_right_expected_label + 1

    bridge_point = hourglass[[3]]

    incdbscan4.insert(top_right)
    incdbscan4.insert(bridge_point)
    incdbscan4.insert(bottom_right)

    assert_cluster_labels(incdbscan4, top_right, top_right_expected_label)
    assert_cluster_labels(
        incdbscan4, bottom_right, bottom_right_expected_label)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point,
        {bottom_right_expected_label, bottom_right_expected_label}
    )

    merged_cluster_expected_label = \
        incdbscan4.get_cluster_labels(bridge_point)[0]

    # Insert objects to the left
    left_pre_cluster = np.array([
        [-EPS, 0],
        [-EPS * 2, 0],
        [-EPS * 2, 0],
    ])
    left_cluster_expected_label = bottom_right_expected_label + 1

    insert_objects_then_assert_cluster_labels(
        incdbscan4,
        left_pre_cluster,
        CLUSTER_LABEL_NOISE
    )

    # Insert object to the center
    new_object = point_at_origin
    incdbscan4.insert(new_object)

    assert_cluster_labels(
        incdbscan4, top_right, merged_cluster_expected_label)
    assert_cluster_labels(
        incdbscan4, bottom_right, merged_cluster_expected_label)
    assert_cluster_labels(
        incdbscan4, bridge_point, merged_cluster_expected_label)
    assert_cluster_labels(
        incdbscan4, left_pre_cluster, left_cluster_expected_label)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        new_object,
        {merged_cluster_expected_label, left_cluster_expected_label}
    )


def test_two_mergers_can_happen_at_the_same_time(
        incdbscan4,
        point_at_origin,
        hourglass_on_the_right):

    # Insert objects to the right
    top_right = hourglass_on_the_right[:3]
    top_right_expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    bottom_right = hourglass_on_the_right[-3:]
    bottom_right_expected_label = top_right_expected_label + 1

    bridge_point_right = hourglass_on_the_right[[3]]

    incdbscan4.insert(top_right)
    incdbscan4.insert(bridge_point_right)
    incdbscan4.insert(bottom_right)

    assert_cluster_labels(incdbscan4, top_right, top_right_expected_label)
    assert_cluster_labels(
        incdbscan4, bottom_right, bottom_right_expected_label)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point_right,
        {bottom_right_expected_label, bottom_right_expected_label}
    )

    # Insert objects to the left
    hourglass_on_the_left = reflect_horizontally(hourglass_on_the_right)

    top_left = hourglass_on_the_left[:3]
    top_left_expected_label = bottom_right_expected_label + 1

    bottom_left = hourglass_on_the_left[-3:]
    bottom_left_expected_label = top_left_expected_label + 1

    bridge_point_left = hourglass_on_the_left[[3]]

    incdbscan4.insert(top_left)
    incdbscan4.insert(bridge_point_left)
    incdbscan4.insert(bottom_left)

    assert_cluster_labels(incdbscan4, top_left, top_left_expected_label)
    assert_cluster_labels(incdbscan4, bottom_left, bottom_left_expected_label)

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point_left,
        {top_left_expected_label, bottom_left_expected_label}
    )

    # Insert object to the center
    new_object = point_at_origin
    incdbscan4.insert(new_object)

    assert_cluster_labels(
        incdbscan4,
        np.vstack([top_right, bottom_right]),
        bottom_right_expected_label
    )

    assert_cluster_labels(
        incdbscan4,
        np.vstack([top_left, bottom_left]),
        bottom_left_expected_label
    )

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point_right,
        {bottom_left_expected_label, bottom_right_expected_label}
    )

    assert_label_of_object_is_among_possible_ones(
        incdbscan4,
        bridge_point_left,
        {top_left_expected_label, bottom_left_expected_label}
    )


def test_object_is_core_if_it_has_more_than_enough_neighors(
        incdbscan3,
        point_at_origin):

    neigbors = np.array([
        [0, EPS],
        [0, -EPS],
        [EPS, 0],
        [-EPS, 0],
    ])
    expected_label = CLUSTER_LABEL_FIRST_CLUSTER

    incdbscan3.insert(neigbors)
    incdbscan3.insert(point_at_origin)

    assert_cluster_labels(incdbscan3, neigbors, expected_label)
    assert_cluster_labels(incdbscan3, point_at_origin, expected_label)
