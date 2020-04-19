import pytest

from incdbscan import IncrementalDBSCANWarning


def test_no_warning_raised_if_unknown_id_is_passed_to_insert(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle

    with pytest.warns(None) as record:
        incdbscan4.insert_objects(object_values, object_ids)

    number_of_warnings = len(record)
    assert number_of_warnings == 0


def test_warning_raised_if_known_id_is_passed_to_insert(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle
    incdbscan4.insert_objects(object_values, object_ids)

    with pytest.warns(IncrementalDBSCANWarning):
        incdbscan4.insert_objects([object_values[0]], [object_ids[0]])


def test_warning_raised_if_unknown_id_is_passed_to_delete(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle
    incdbscan4.insert_objects(object_values, object_ids)

    unknown_index = 'UNKNOWN'

    with pytest.warns(IncrementalDBSCANWarning):
        one_known_one_unknown = [object_ids[0], unknown_index]
        incdbscan4.delete_objects(one_known_one_unknown)


def test_no_warning_raised_if_known_index_is_passed_to_delete(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle
    incdbscan4.insert_objects(object_values, object_ids)

    with pytest.warns(None) as record:
        incdbscan4.delete_objects(object_ids[1:5])

    number_of_warnings = len(record)
    assert number_of_warnings == 0


def test_labels_are_accessible_for_inserted_objects(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle
    incdbscan4.insert_objects(object_values, object_ids)

    for object_id in object_ids:
        assert object_id in incdbscan4.labels


def test_labels_are_not_accessible_for_not_inserted_objects(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle
    incdbscan4.insert_objects(object_values, object_ids)

    unknown_id = 'UNKNOWN'
    assert unknown_id not in incdbscan4.labels


def test_labels_are_not_accessible_for_deleted_objects(
        incdbscan4,
        blob_in_middle):

    object_values, object_ids = blob_in_middle
    incdbscan4.insert_objects(object_values, object_ids)

    first_object_id = object_ids[0]
    incdbscan4.delete_objects([first_object_id])

    assert first_object_id not in incdbscan4.labels
