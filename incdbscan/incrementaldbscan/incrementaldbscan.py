import warnings

import numpy as np

from ._deleter import _Deleter
from ._inserter import _Inserter
from ._objects import _Objects
from ._utils import input_check


class IncrementalDBSCAN:
    """The incremental version of DBSCAN, a density-based clustering algorithm
    that handles outliers.

    After an initial clustering of an initial object set (i.e., set of data
    points), the object set can at any time be updated by increments of any
    size. An increment can be either the insertion or the deletion of objects.

    After each update, the result of the clustering is the same as if the
    updated object set (i.e., the initial object set modified by all of the
    increments) was clustered by DBSCAN. However, this result is reached by
    using information from the previous state of the clustering, and without
    the need of applying DBSCAN to the whole updated object set. Therefore, it
    is more efficient.

    Parameters
    ----------
    eps : float, default=0.5
        The radius of neighborhood calculation. An object is the neighbor of
        another if the distance between them is no more than eps.

    min_pts : int, default=5
        The minimum number of neighbors that an object needs to have to be a
        core object of a cluster.

    References
    ----------
    Ester et al. 1998. Incremental Clustering for Mining in a Data Warehousing
    Environment. In: Proceedings of the 24rd International Conference on Very
    Large Data Bases (VLDB 1998).

    """

    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts

        self._objects = _Objects(self.eps)
        self._inserter = _Inserter(self.eps, self.min_pts, self._objects)
        self._deleter = _Deleter(self.eps, self.min_pts, self._objects)

    def insert(self, X):
        """Insert objects into the object set, then update clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to be inserted into the object set.

        Returns
        -------
        self

        """
        X = input_check(X)

        for value in X:
            self._inserter.insert(value)

        return self

    def delete(self, X):
        """Delete objects from object set, then update clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to be deleted from the object set.

        Returns
        -------
        self

        """
        X = input_check(X)

        for ix, value in enumerate(X):
            obj = self._objects.get_object(value)

            if obj:
                self._deleter.delete(obj)

            else:
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'Object at position {ix} was not deleted because '
                        'there is no such object in the object set.'
                    )
                )

        return self

    def get_cluster_labels(self, X):
        """Get cluster labels of objects.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data objects to get labels for.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
                 Cluster labels. Effective labels start from 0. -1 means the
                 object is noise. numpy.nan means the object was not in the
                 object set.

        """
        X = input_check(X)

        labels = np.zeros(len(X))

        for ix, value in enumerate(X):
            obj = self._objects.get_object(value)
            label = obj.label if obj else np.nan
            labels[ix] = label

            if np.isnan(label):
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'No label was retrieved for object at position {ix} '
                        'because there is no such object in the object set.'
                    )
                )

        return labels


class IncrementalDBSCANWarning(Warning):
    pass


# TODO metrics in arguments
# TODO: initial fitting

# ALGO related
# TODO functional tests

# UX related
# TODO readme: API usage
# TODO gif example
# TODO notebook: example

# Performance relatedgit
# TODO indexing: use KDTree for initial tree building
# TODO readme: performance

# Packaging related
# TODO pip upload
# TODO readme: installation
