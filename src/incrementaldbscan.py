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
        self.neighbor_count = 1


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
        for neighbor in neighbors:
            neighbor.neighbor_count += 1

        update_seeds = self._get_update_seeds_during_insertion(neighbors)

        # Noise
        if not update_seeds:
            self.labels[object_to_insert.id] = self.CLUSTER_LABEL_NOISE
            return

        FAKE_CLUSTER_ID = 888  # TODO remove faking
        self.labels[object_to_insert.id] = FAKE_CLUSTER_ID

    def _deletion(self, object_id):
        # Do lot of stuff
        del self.labels[object_id]

    def _get_neighbors(self, query_object, only_cores=False):
        return [
            obj for obj in self._object_store.values()
            if euclidean_distance(query_object.value, obj.value) <= self.eps
            and (not only_cores or obj.neighbor_count >= self.min_pts)
        ]

    def _get_update_seeds_during_insertion(self, neighbors):
        update_seeds = set()

        new_core_objects = [obj for obj in neighbors
                            if obj.neighbor_count == self.min_pts]
        update_seeds.update(new_core_objects)

        for new_core in new_core_objects:
            # TODO this can be more efficient as described by the paper
            core_neighbors_of_new_core = \
                self._get_neighbors(new_core, only_cores=True)
            update_seeds.update(core_neighbors_of_new_core)

        return update_seeds


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


class IncrementalDBSCANWarning(Warning):
    pass
