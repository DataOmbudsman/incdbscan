import warnings
from typing import Dict, Iterable

from src.incrementaldbscan._deleter import _Deleter
from src.incrementaldbscan._inserter import _Inserter
from src.incrementaldbscan._labels import ClusterLabel, _Labels
from src.incrementaldbscan._objects import _Object, ObjectId, _Objects


class IncrementalDBSCAN:
    """An incremental density-based clustering algorithm that handles outliers.

    After an initial clustering of objects, the clustering can at any time be
    updated by increments of any size. An increment can be either addition or
    or deletion of objects.

    After each update the result of the clustering is the same as if the
    updated object set (i.e., the initial object set modified by the
    increments) was clustered by DBSCAN. However, this result is reached by
    using information from the previous state of clustering, and without the
    need of applying DBSCAN to the updated object set. Therefore, it is more
    efficient.

    Parameters
    ----------
    eps : float, default=0.5
        The radius of neighborhood calculation. An object is the neighbor of
        another if the distance between them is no more than eps.

    min_pts : int, default=5
        The minimum number of neighbors that an object needs to have to be a
        core object of a cluster.

    cache_size : int, default=256
        Size of cache for caching neighbor retrieval within an update. The
        larger the value, the faster the calculation is, at the expense of
        memory need.

    Attributes
    ----------
    labels : dict
        Cluster label for each object, stored as a dictionary mapping
        object IDs to cluster labels. Label -1 means noise.

    References
    ----------
    Ester et al. 1998. Incremental Clustering for Mining in a Data Warehousing
    Environment. In: Proceedings of the 24rd International Conference on Very
    Large Data Bases (VLDB 1998).

    """

    def __init__(self, eps=0.5, min_pts=5, cache_size=256):
        self.eps = eps
        self.min_pts = min_pts
        self.cache_size = cache_size

        self._labels = _Labels()
        self._objects = _Objects(self.cache_size)

        self._inserter = _Inserter(
            self.eps, self.min_pts, self._labels, self._objects)
        self._deleter = _Deleter(
            self.eps, self.min_pts, self._labels, self._objects)

    @property
    def labels(self) -> Dict[ObjectId, ClusterLabel]:
        return self._labels.get_all_labels()

    def add_object(self, object_value, object_id: ObjectId):
        """Add object to clustering.

        Parameters
        ----------
        object_value : array
            A numpy array representing the data object to be added to the
            clustering.

        object_id : str or int
            The identifier of the data object to be added to the clustering.

        """

        if object_id not in self.labels:
            object_to_insert = _Object(object_value, object_id)
            self._inserter.insert(object_to_insert)

        else:
            warnings.warn(
                IncrementalDBSCANWarning(
                    f'Object with ID {object_id} was not added '
                    'because it already exists in the clustering.'
                )
            )

    def add_objects(
            self,
            object_values: Iterable,
            object_ids: Iterable[ObjectId]):
        """Add objects to clustering.

        Parameters
        ----------
        object_values : array
            An array of numpy arrays, representing the data objects to be added
            to the clustering.

        object_ids : iterable of int or of str
            The identifiers of the data objects to be added to the clustering.
            E.g., list of strings, or numpy array of integers.

        """
        for object_value, object_id in zip(object_values, object_ids):
            self.add_object(object_value, object_id)

    def delete_objects(self, object_ids: Iterable):
        """Delete objects from object set, then update clustering.

        Parameters
        ----------
        object_ids : iterable of int or of str
            The identifiers of the data objects to be deleted from the object
            set. E.g., list of strings, or numpy array of integers.

        """
        for object_id in object_ids:

            if object_id in self.labels:
                self._deleter.delete(object_id)

            else:
                warnings.warn(
                    IncrementalDBSCANWarning(
                        f'Object with ID {object_id} was not deleted '
                        f'because there is no object with this ID.'
                    )
                )


class IncrementalDBSCANWarning(Warning):
    pass


# TODO metrics in arguments
# TODO make API more sklearn-like.
    # Step 1: 1 insert method, 1 delete method
    # Step 2: no ID argument, use hash instead
    # Step 3: create predict
    # Step 4: 1 update method -> partial_fit
    # Step 5: fit as initial fitting

# TODO ID type only int?
# TODO "insert" api doc change

# ALGO related
# TODO functional tests
# TODO remove prints

# UX related
# TODO readme: intro
# TODO readme: API usage
# TODO gif example
# TODO notebook: example

# Performance related
# TODO validate purpose of cache
# TODO indexing: use KDTree for initial tree building
# TODO indexing: cKDTree
# TODO readme: performance

# Packaging related
# TODO __init__ file setup
# TODO package restructure: no src
# TODO pip
# TODO readme: installation
