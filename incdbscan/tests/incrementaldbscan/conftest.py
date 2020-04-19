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
    values, _ = make_blobs(
        n_samples=10,
        centers=[[0, 0]],
        n_features=2,
        cluster_std=0.4,
        random_state=123
    )
    ids = np.arange(len(values))
    return values, ids


@pytest.fixture
def object_far_away():
    value = np.array([[10, 10]])
    id_ = 'FARAWAY'
    return value, id_


@pytest.fixture
def point_at_origin():
    value = np.array([[0, 0]])
    id_ = 'NEW'
    return value, id_


@pytest.fixture
def three_points_on_the_left():
    values = np.array([
        [-EPS, 0],
        [-EPS * 2, 0],
        [-EPS * 3, 0],
    ])
    ids = ['left0', 'left1', 'left2']
    return values, ids


@pytest.fixture
def three_points_on_the_top():
    values = np.array([
        [0, EPS],
        [0, EPS * 2],
        [0, EPS * 3],
    ])
    ids = ['top0', 'top1', 'top2']
    return values, ids


@pytest.fixture
def three_points_at_the_bottom():
    values = np.array([
        [0, -EPS],
        [0, -EPS * 2],
        [0, -EPS * 3],
    ])
    ids = ['bottom0', 'bottom1', 'bottom2']
    return values, ids


@pytest.fixture
def hourglass_on_the_right():
    values = np.array([
        [EPS, EPS * 2],
        [EPS, EPS * 2],
        [EPS, EPS],
        [EPS, 0],
        [EPS, -EPS],
        [EPS, -EPS * 2],
        [EPS, -EPS * 2],
    ])
    ids = [100, 200, 300, 400, 500, 600, 700]
    return values, ids
