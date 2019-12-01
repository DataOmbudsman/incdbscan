import warnings
from typing import Dict, Iterable, Union

from src.dbscanbase import DBSCANBase


ClusterId = int
ObjectId = Union[int, str]


class IncrementalDBSCAN(DBSCANBase):
    """
    Based on "Incremental Clustering for Mining in a Data Warehousing
    Environment by Ester et al. 1998.
    """

    def __init__(self, eps, min_pts):
        super().__init__(eps, min_pts)
        self.labels: Dict[ObjectId, ClusterId] = dict()

    def add_object(self, object_, object_id: ObjectId):
        """
        Add object, along with its ID, to clustering.
        ID is of an unhashable type, such as int or str.
        """

        if object_id not in self.labels:
            self._update_clustering_due_to_insertion(object_, object_id)

        else:
            warnings.warn(
                IncrementalDBSCANWarning(
                    f'Object with ID {object_id} was not added '
                    'because it already exists in the clustering.'
                )
            )

    def add_objects(self, objects: Iterable, object_ids: Iterable[ObjectId]):
        """
        Add objects, along with their IDs, to clustering.
        ID is of an unhashable type, such as int or str.
        """

        for object_, object_id in zip(objects, object_ids):
            self.add_object(object_, object_id)

    def remove_object(self, object_id: ObjectId):
        """Remove object, given by its ID, from clustering."""

        if object_id in self.labels:
            self._update_clustering_due_to_deletion(object_id)

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

    def _update_clustering_due_to_insertion(self, object_, object_id):
        # TODO interface?
        FAKE_CLUSTER_ID = 888
        self.labels[object_id] = FAKE_CLUSTER_ID  # TODO remove faking

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)
    def _update_clustering_due_to_deletion(self, object_id):
        _ = self.labels.pop(object_id)


class IncrementalDBSCANWarning(Warning):
    pass
