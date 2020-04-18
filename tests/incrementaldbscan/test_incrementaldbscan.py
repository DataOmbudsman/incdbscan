import pytest

from src.incrementaldbscan.incrementaldbscan import (
    IncrementalDBSCAN,
    IncrementalDBSCANWarning
)


@pytest.fixture
def incdbscan():
    return IncrementalDBSCAN(eps=0.5, min_pts=5)


@pytest.fixture
def inputs():
    object_values = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    object_ids = range(len(object_values))
    return object_values, object_ids


def test_no_warning_raised_if_unknown_id_is_passed_to_add(incdbscan, inputs):
    object_values, object_ids = inputs

    with pytest.warns(None) as record:
        incdbscan.add_objects(object_values, object_ids)

    number_of_warnings = len(record)
    assert number_of_warnings == 0


def test_warning_raised_if_known_id_is_passed_to_add(incdbscan, inputs):
    object_values, object_ids = inputs
    incdbscan.add_objects(object_values, object_ids)

    with pytest.warns(IncrementalDBSCANWarning):
        incdbscan.add_objects([object_values[0]], [object_ids[0]])


def test_warning_raised_if_unknown_id_is_passed_to_delete(incdbscan, inputs):
    object_values, object_ids = inputs
    incdbscan.add_objects(object_values, object_ids)

    unknown_index = 'UNKNOWN'

    with pytest.warns(IncrementalDBSCANWarning):
        one_known_one_unknown = [object_ids[0], unknown_index]
        incdbscan.delete_objects(one_known_one_unknown)


def test_no_warning_raised_if_known_index_is_passed_to_delete(
        incdbscan,
        inputs):

    object_values, object_ids = inputs
    incdbscan.add_objects(object_values, object_ids)

    with pytest.warns(None) as record:
        incdbscan.delete_objects(object_ids[1:5])

    number_of_warnings = len(record)
    assert number_of_warnings == 0


def test_labels_are_accessible_for_added_objects(incdbscan, inputs):
    object_values, object_ids = inputs
    incdbscan.add_objects(object_values, object_ids)

    for object_id in object_ids:
        assert object_id in incdbscan.labels


def test_labels_are_not_accessible_for_not_added_objects(incdbscan, inputs):
    object_values, object_ids = inputs
    incdbscan.add_objects(object_values, object_ids)

    unknown_id = 'UNKNOWN'
    assert unknown_id not in incdbscan.labels


def test_labels_are_not_accessible_for_deleted_objects(incdbscan, inputs):
    object_values, object_ids = inputs
    incdbscan.add_objects(object_values, object_ids)

    first_object_id = object_ids[0]
    incdbscan.delete_objects([first_object_id])

    assert first_object_id not in incdbscan.labels
