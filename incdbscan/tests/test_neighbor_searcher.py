import numpy as np
import pytest

from incdbscan._neighbor_searcher import (
    NeighborSearcher,
    _CKDTreeNeighborSearcher,
    _SklearnNeighborSearcher
)


METRICS_PS = [
    ('minkowski', 0.5),
    ('minkowski', 1),
    ('minkowski', 2),
    ('minkowski', 3),
    ('minkowski', np.inf),
    ('p', 0.5),
    ('p', 1),
    ('p', 2),
    ('p', 3),
    ('p', np.inf),
    ('manhattan', None),
    ('cityblock', None),
    ('l1', None),
    ('euclidean', None),
    ('l2', None),
    ('chebyshev', None),
    ('infinity', None),
]

@pytest.mark.parametrize("metric, p", METRICS_PS)
def test_alias_normalization_through_neighbor_searcher(metric, p):
    points_low_dimension = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [5.0, 5.0],
    ])
    high_dimension_count = 20
    pad = np.zeros((len(points_low_dimension), high_dimension_count))
    points_high_dimension = np.hstack([points_low_dimension, pad])
    points_zipped = list(zip(points_low_dimension, points_high_dimension))

    radius = 1.5
    low_searcher = NeighborSearcher(radius, metric, p)
    high_searcher = NeighborSearcher(radius, metric, p)

    for i, (point_low, point_high) in enumerate(points_zipped):
        low_searcher.insert(point_low, new_id=i)
        high_searcher.insert(point_high, new_id=i)

    for point_low, point_high in points_zipped:
        assert sorted(low_searcher.query_neighbors(point_low)) == \
               sorted(high_searcher.query_neighbors(point_high))


def test_p_less_than_1_falls_back_to_sklearn():
    neighbor_searcher = NeighborSearcher(radius=1.0, metric='minkowski', p=0.5)
    neighbor_searcher.insert(np.zeros(5), 0)
    assert isinstance(
        neighbor_searcher._effective_searcher, _SklearnNeighborSearcher)


def test_high_dim_falls_back_to_sklearn():
    neighbor_searcher = NeighborSearcher(radius=1.0, metric='euclidean', p=2)
    neighbor_searcher.insert(np.zeros(20), 0)
    assert isinstance(
        neighbor_searcher._effective_searcher, _SklearnNeighborSearcher)


def test_low_dim_uses_ckdtree():
    neighbor_searcher = NeighborSearcher(radius=1.0, metric='euclidean', p=2)
    neighbor_searcher.insert(np.zeros(5), 0)
    assert isinstance(
        neighbor_searcher._effective_searcher, _CKDTreeNeighborSearcher)
