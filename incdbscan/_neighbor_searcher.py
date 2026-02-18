import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sortedcontainers import SortedList


_HIGH_DIM_THRESHOLD = 15

_MINKOWSKI_P = {
    'minkowski': None,
    'p': None,
    'manhattan': 1,
    'cityblock': 1,
    'l1': 1,
    'euclidean': 2,
    'l2': 2,
    'chebyshev': float('inf'),
    'infinity': float('inf'),
}


class _BaseNeighborSearcher:
    def __init__(self):
        self.values = np.array([])
        self.ids = SortedList()

    def insert(self, new_value, new_id):
        self.ids.add(new_id)
        position = self.ids.index(new_id)

        self._insert_into_array(new_value, position)
        self._rebuild()

    def _insert_into_array(self, new_value, position):
        extended = np.insert(self.values, position, new_value, axis=0)
        if not self.values.size:
            extended = extended.reshape(1, -1)
        self.values = extended

    def _rebuild(self):
        raise NotImplementedError

    def _get_neighbor_indices(self, query_value):
        raise NotImplementedError

    def query_neighbors(self, query_value):
        neighbor_indices = self._get_neighbor_indices(query_value)

        for ix in neighbor_indices:
            yield self.ids[ix]

    def delete(self, id_):
        position = self.ids.index(id_)
        del self.ids[position]
        self.values = np.delete(self.values, position, axis=0)


class _CKDTreeNeighborSearcher(_BaseNeighborSearcher):
    def __init__(self, radius, p):
        super().__init__()
        self.radius = radius
        self.p = p
        self._tree = None

    def _rebuild(self):
        self._tree = cKDTree(self.values)

    def _get_neighbor_indices(self, query_value):
        return self._tree.query_ball_point(
            query_value, self.radius, p=self.p)


class _SklearnNeighborSearcher(_BaseNeighborSearcher):
    def __init__(self, radius, metric, p):
        super().__init__()
        self._nn = NearestNeighbors(radius=radius, metric=metric, p=p)

    def _rebuild(self):
        self._nn = self._nn.fit(self.values)

    def _get_neighbor_indices(self, query_value):
        return self._nn.radius_neighbors(
            [query_value], return_distance=False)[0]


class NeighborSearcher:
    # For non-Minkowski metrics use sklearn's deafult NeighborSearcher.
    # For Minkowski metrics, cKDTree is faster but suffers from high
    # dimensionality and doesn't support p < 1. Here, decision between cKDTree
    # and sklearn is deferred until the first insert reveals dimensionality.

    def __init__(self, radius, metric, p):
        self._radius = radius
        self._metric = metric
        self._p = p
        self._effective_p = None
        self._effective_searcher = None

    def _pick_effective_searcher(self, new_value):
        if self._metric in _MINKOWSKI_P:
            self._effective_p = (self._p if _MINKOWSKI_P[self._metric] is None
                                 else _MINKOWSKI_P[self._metric])

            low_dimensional = len(new_value) <= _HIGH_DIM_THRESHOLD
            if self._effective_p >= 1 and low_dimensional:
                self._effective_searcher = _CKDTreeNeighborSearcher(
                    self._radius, self._effective_p)
            else:
                self._effective_searcher = _SklearnNeighborSearcher(
                    self._radius, 'minkowski', self._effective_p)
        else:
            self._effective_p = None
            self._effective_searcher = _SklearnNeighborSearcher(
                self._radius, self._metric, self._p)

    def insert(self, new_value, new_id):
        if self._effective_searcher is None:
            self._pick_effective_searcher(new_value)
        self._effective_searcher.insert(new_value, new_id)

    def query_neighbors(self, query_value):
        return self._effective_searcher.query_neighbors(query_value)

    def delete(self, id_):
        self._effective_searcher.delete(id_)
