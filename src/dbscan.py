import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.dbscanbase import _DBSCANBase


class DBSCAN(_DBSCANBase):
    """
    Based on "A Density-Based Algorithm for Discovering Clusters in Large
    Spatial Databases with Noise" by Ester et al. 1996.
    """

    def __init__(self, eps, min_pts):
        super().__init__(eps, min_pts)

    def _init_fit(self, X):
        self.labels = np.repeat([DBSCAN.CLUSTER_LABEL_UNCLASSIFIED], len(X))
        self._nn = NearestNeighbors(radius=self.eps).fit(X)

    def _is_point_unclassified(self, ix):
        return self.labels[ix] == DBSCAN.CLUSTER_LABEL_UNCLASSIFIED

    def _is_point_noise(self, ix):
        return self.labels[ix] == DBSCAN.CLUSTER_LABEL_NOISE

    def _assign_label(self, ix, label):
        self.labels[ix] = label

    def _get_neighbors(self, query_point):
        neighbors = \
            self._nn.radius_neighbors([query_point], return_distance=False)[0]
        return set(neighbors)

    def _expand_cluster(self, ix, X):
        point = X[ix]
        seeds = self._get_neighbors(point)

        if len(seeds) < self.min_pts:
            self._assign_label(ix, DBSCAN.CLUSTER_LABEL_NOISE)
            return

        for seed in seeds:
            self._assign_label(seed, self._next_cluster_label)

        seeds.remove(ix)

        while len(seeds) > 0:
            seed_point = X[seeds.pop()]  # called currentP in paper
            neighbors_of_seed = self._get_neighbors(seed_point)

            if len(neighbors_of_seed) >= self.min_pts:
                for neighbor in neighbors_of_seed:
                    if self._is_point_unclassified(neighbor):
                        seeds.add(neighbor)
                        self._assign_label(neighbor, self._next_cluster_label)
                    elif self._is_point_noise(neighbor):
                        self._assign_label(neighbor, self._next_cluster_label)

        self._next_cluster_label += 1

    def fit(self, X):
        self._init_fit(X)

        for ix in range(len(X)):
            if self._is_point_unclassified(ix):
                self._expand_cluster(ix, X)

        return self
