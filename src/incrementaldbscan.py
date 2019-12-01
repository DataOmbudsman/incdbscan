import numpy as np
import warnings
from typing import Dict, Iterable, Union

from src.dbscanbase import DBSCANBase


ClusterId = int
ObjectId = Union[int, str]


class Object():
    def __init__(self, object_value, object_id):
        self.value = object_value
        self.id = object_id


class IncrementalDBSCAN(DBSCANBase):
    """
    Based on "Incremental Clustering for Mining in a Data Warehousing
    Environment by Ester et al. 1998.
    """

    def __init__(self, eps, min_pts):
        super().__init__(eps, min_pts)
        self.labels: Dict[ObjectId, ClusterId] = dict()
        self._object_store: Dict[ObjectId, Object] = dict()

    def add_object(self, object_value, object_id: ObjectId):
        """
        Add an object (given by its value), along with its ID, to clustering.
        ID is of an unhashable type, such as int or str.
        """

        if object_id not in self.labels:
            object_to_insert = Object(object_value, object_id)
            self._insertion(object_to_insert)

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
            self._deletion(object_id)

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

    def _insertion(self, object_to_insert: Object):
        self._object_store[object_to_insert.id] = object_to_insert

        neighbors = self._get_neighbors(object_to_insert)
        # TODO eldönteni milyen típusból mennyi van

        FAKE_CLUSTER_ID = 888  # TODO remove faking
        self.labels[object_to_insert.id] = FAKE_CLUSTER_ID

    def _deletion(self, object_id):
        # Do lot of stuff
        del self.labels[object_id]

    def _get_neighbors(self, query_object):
        return [
            obj for obj in self._object_store.values()
            if euclidean_distance(query_object.value, obj.value) <= self.eps
        ]


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


class IncrementalDBSCANWarning(Warning):
    pass
