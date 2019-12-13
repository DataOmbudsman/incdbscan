import warnings
from typing import Dict, Iterable

from src.incrementaldbscan._labels import ClusterId, _Labels
from src.incrementaldbscan._objects import _Object, _Objects, ObjectId
from src.incrementaldbscan._updater import _Updater
from src.dbscan._dbscanbase import _DBSCANBase


class IncrementalDBSCAN(_DBSCANBase):
    """
    Based (mostly but not completely) on "Incremental Clustering for Mining
    in a Data Warehousing Environment" by Ester et al. 1998.
    """

    def __init__(self, eps, min_pts):
        super().__init__(eps, min_pts)
        self._labels = _Labels()
        self._objects = _Objects()
        self._updater = _Updater(self)

        self.labels: Dict[ObjectId, ClusterId] = self._labels.get_all_labels()

    def add_object(self, object_value, object_id: ObjectId):
        """
        Add an object (given by its value), along with its ID, to clustering.
        ID is of an unhashable type, such as int or str.
        """

        if not self._labels.has_label(object_id):
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

        if self._labels.has_label(object_id):
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
# TODO pipenv
# TODO make file?
# TODO __init__ file setup?
