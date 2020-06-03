import numpy as np
from sklearn.neighbors import NearestNeighbors
from sortedcontainers import SortedList


class NeighborSearcher:
    def __init__(self, radius, metric, p):
        self.neighbor_searcher = \
            NearestNeighbors(radius=radius, metric=metric, p=p)
        self.values = np.array([])
        self.ids = SortedList()

    def insert(self, new_value, new_id):
        self.ids.add(new_id)
        position = self.ids.index(new_id)

        self._insert_into_array(new_value, position)
        self.neighbor_searcher = self.neighbor_searcher.fit(self.values)

    def _insert_into_array(self, new_value, position):
        extended = np.insert(self.values, position, new_value, axis=0)
        if not self.values.size:
            extended = extended.reshape(1, -1)
        self.values = extended

    def query_neighbors(self, query_value):
        neighbor_indices = self.neighbor_searcher.radius_neighbors(
            [query_value], return_distance=False)[0]

        for ix in neighbor_indices:
            yield self.ids[ix]

    def delete(self, id_):
        position = self.ids.index(id_)
        del self.ids[position]
        self.values = np.delete(self.values, position, axis=0)
