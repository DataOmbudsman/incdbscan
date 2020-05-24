from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from incdbscan import IncrementalDBSCAN, IncrementalDBSCANWarning
from testutils import (
    are_lists_isomorphic,
    CLUSTER_LABEL_NOISE,
    delete_object_and_assert_error,
    delete_object_and_assert_no_warning,
    delete_object_and_assert_warning,
    insert_object_and_assert_error,
    insert_objects_then_assert_cluster_labels,
    get_label_and_assert_error,
    get_label_and_assert_no_warning,
    get_label_and_assert_warning
)


def test_error_when_input_is_non_numeric(incdbscan3):
    inputs_not_welcomed = np.array([
        [1, 2, 'x'],
        [1, 2, None],
        [1, 2, np.nan],
        [1, 2, np.inf],
    ])

    for i in range(len(inputs_not_welcomed)):
        input_ = inputs_not_welcomed[[i]]

        insert_object_and_assert_error(incdbscan3, input_, ValueError)
        delete_object_and_assert_error(incdbscan3, input_, ValueError)
        get_label_and_assert_error(incdbscan3, input_, ValueError)


def test_handling_of_same_object_with_different_dtype(incdbscan3):
    object_as_int = np.array([[1, 2]])
    object_as_float = np.array([[1., 2.]])

    incdbscan3.insert(object_as_int)

    assert incdbscan3.get_cluster_labels(object_as_int) == \
        incdbscan3.get_cluster_labels(object_as_float)

    delete_object_and_assert_no_warning(incdbscan3, object_as_float)


def test_handling_of_more_than_2d_arrays(incdbscan3):
    object_3d = np.array([[1, 2, 3]])
    object_100d = np.random.random(100).reshape(1, -1)

    incdbscan3.insert(object_3d)
    incdbscan3.insert(object_3d)
    incdbscan3.delete(object_3d)

    assert incdbscan3.get_cluster_labels(object_3d) == CLUSTER_LABEL_NOISE

    incdbscan3.delete(object_3d)
    incdbscan3.insert(object_100d)
    incdbscan3.insert(object_100d)
    incdbscan3.delete(object_100d)

    assert incdbscan3.get_cluster_labels(object_100d) == CLUSTER_LABEL_NOISE


def test_no_warning_when_a_known_object_is_deleted(
        incdbscan3,
        point_at_origin):

    incdbscan3.insert(point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, point_at_origin)

    incdbscan3.insert(point_at_origin)
    incdbscan3.insert(point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, point_at_origin)
    delete_object_and_assert_no_warning(incdbscan3, point_at_origin)


def test_warning_when_unknown_object_is_deleted(
        incdbscan3,
        point_at_origin,
        object_far_away):

    delete_object_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)

    incdbscan3.insert(point_at_origin)

    incdbscan3.delete(point_at_origin)

    delete_object_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)


def test_no_warning_when_cluster_label_is_gotten_for_known_object(
        incdbscan3,
        point_at_origin):

    expected_label = np.array([CLUSTER_LABEL_NOISE])

    incdbscan3.insert(point_at_origin)
    label = get_label_and_assert_no_warning(incdbscan3, point_at_origin)
    assert label == expected_label

    incdbscan3.insert(point_at_origin)
    incdbscan3.delete(point_at_origin)
    label = get_label_and_assert_no_warning(incdbscan3, point_at_origin)
    assert label == expected_label


def test_warning_when_cluster_label_is_gotten_for_unknown_object(
        incdbscan3,
        point_at_origin,
        object_far_away):

    label = get_label_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)
    assert np.isnan(label)

    incdbscan3.insert(point_at_origin)
    incdbscan3.delete(point_at_origin)

    label = get_label_and_assert_warning(
        incdbscan3, point_at_origin, IncrementalDBSCANWarning)
    assert np.isnan(label)


def test_different_metrics_are_available():
    incdbscan_euclidean = \
        IncrementalDBSCAN(eps=1.5, min_pts=3, metric='euclidean')
    incdbscan_manhattan = \
        IncrementalDBSCAN(eps=1.5, min_pts=3, metric='manhattan')

    diagonal = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
    ])

    expected_label_euclidean = CLUSTER_LABEL_NOISE + 1
    insert_objects_then_assert_cluster_labels(
        incdbscan_euclidean, diagonal, expected_label_euclidean)

    expected_label_manhattan = CLUSTER_LABEL_NOISE
    insert_objects_then_assert_cluster_labels(
        incdbscan_manhattan, diagonal, expected_label_manhattan)


@pytest.mark.slow
def test_same_results_as_sklearn_dbscan():
    base_path = Path(__file__).parent
    data_path = base_path / 'data' / '2d-20c-no0.dat'
    data = np.loadtxt(data_path)[:, 0:2]

    EPS = 1
    MIN_PTS = 5

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan_1 = incdbscan.insert(data).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_1)

    labels_incdbscan_2 = \
        incdbscan.insert(data).delete(data).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_2)

    np.random.seed(123)
    noise = np.random.uniform(-14, 14, (1000, 2))
    labels_incdbscan_3 = \
        incdbscan.insert(noise).delete(noise).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_3)
