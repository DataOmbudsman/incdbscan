import warnings
from typing import Dict, Iterable

from src.incrementaldbscan._labels import ClusterLabel
from src.incrementaldbscan._objects import _Object, ObjectId
from src.incrementaldbscan._updater import _Updater


class IncrementalDBSCAN():
    """
    Based (mostly but not completely) on "Incremental Clustering for Mining
    in a Data Warehousing Environment" by Ester et al. 1998.
    """

    def __init__(self, eps=0.5, min_pts=5, cache_size=256):
        self.eps = eps
        self.min_pts = min_pts
        self.cache_size = cache_size

        self._updater = _Updater(self.eps, self.min_pts, self.cache_size)
        self.labels: Dict[ObjectId, ClusterLabel] = \
            self._updater.labels.get_all_labels()

    def add_object(self, object_value, object_id: ObjectId):
        """
        Add an object (given by its value), along with its ID, to clustering.
        ID is of an unhashable type, such as int or str.
        """

        if object_id not in self.labels:
            object_to_insert = _Object(object_value, object_id)
            self._updater.insertion(object_to_insert)

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
        """
        Add objects (given by their values), along with their IDs, to
        clustering. ID is of an unhashable type, such as int or str.
        """

        for object_value, object_id in zip(object_values, object_ids):
            self.add_object(object_value, object_id)

    def remove_object(self, object_id: ObjectId):
        """Remove object, given by its ID, from clustering."""

        if object_id in self.labels:
            self._updater.deletion(object_id)

        else:
            warnings.warn(
                IncrementalDBSCANWarning(
                    f'Object with ID {object_id} was not removed '
                    f'because there is no object with this ID.'
                )
            )

    def remove_objects(self, object_ids: Iterable):
        """Remove objects, given by their IDs, from clustering."""

        for object_id in object_ids:
            self.remove_object(object_id)


class IncrementalDBSCANWarning(Warning):
    pass

# TODO notebook import fails now
# TODO __init__ file setup?
