import numpy as np
import pytest
from sklearn.datasets import make_blobs

from incdbscan import IncrementalDBSCAN


EPS = 1.5


@pytest.fixture
def incdbscan3():
    return IncrementalDBSCAN(eps=EPS, min_pts=3)


@pytest.fixture
def incdbscan4():
    return IncrementalDBSCAN(eps=EPS, min_pts=4)


@pytest.fixture
def blob_in_middle():
    # pylint: disable=unbalanced-tuple-unpacking
    blob, _ = make_blobs(
        n_samples=10,
        centers=[[0, 0]],
        n_features=2,
        cluster_std=0.4,
        random_state=123,
        return_centers=False
    )
    return blob


@pytest.fixture
def object_far_away():
    return np.array([[10., 10.]])


@pytest.fixture
def point_at_origin():
    return np.array([[0., 0.]])


@pytest.fixture
def three_points_on_the_left():
    return np.array([
        [-EPS, 0],
        [-EPS * 2, 0],
        [-EPS * 3, 0],
    ])


@pytest.fixture
def three_points_on_the_top():
    return np.array([
        [0, EPS],
        [0, EPS * 2],
        [0, EPS * 3],
    ])


@pytest.fixture
def three_points_at_the_bottom():
    return np.array([
        [0, -EPS],
        [0, -EPS * 2],
        [0, -EPS * 3],
    ])


@pytest.fixture
def hourglass_on_the_right():
    return np.array([
        [EPS, EPS * 2],
        [EPS, EPS * 2],
        [EPS, EPS],
        [EPS, 0],
        [EPS, -EPS],
        [EPS, -EPS * 2],
        [EPS, -EPS * 2],
    ])
