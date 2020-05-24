from testutils import are_lists_isomorphic


def test_isomorphism_check_fails_when_different_length():
    assert not are_lists_isomorphic([0, 0], [0, 0, 0])


def test_isomorphism_check_fails_when_different_size_of_value_sets():
    assert not are_lists_isomorphic([0, 0], [0, 1])


def test_isomorphism_check_fails_when_there_is_no_isomorphism():
    assert not are_lists_isomorphic([1, 2, 3, 1], [1, 1, 3, 2])


def test_isomorphism_check_succeeds_when_there_is_isomorphism():
    assert are_lists_isomorphic([1, 1, 2, 2, 3], [2, 2, 3, 3, 4])
