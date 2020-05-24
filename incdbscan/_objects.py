from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ._labels import LabelHandler
from ._object import Object, ObjectId
from ._utils import hash_


class Objects(LabelHandler):
    def __init__(self, eps, metric, p):
        super().__init__()
        self.eps = eps
        self.metric = metric
        self.p = p
        self.objects: Dict[ObjectId, Object] = dict()

        self._neighbor_searcher = \
            NearestNeighbors(radius=eps, metric=self.metric, p=self.p)
        self._values = None
        self._ids = None

    def get_object(self, value):
        id_ = hash_(value)
        return self.objects.get(id_)

    def insert_object(self, value):
        id_ = hash_(value)

        if id_ in self.objects:
            obj = self.objects[id_]
            obj.count += 1
            return obj

        new_object = Object(value, id_)
        self.objects[id_] = new_object
        self.set_label_of_inserted_object(new_object)

        self._fit_neighbor_searcher()
        self._update_neighbors_during_insertion(new_object)
        return new_object

    def _fit_neighbor_searcher(self):
        values = list()
        ids = list()

        for obj in self.objects.values():
            values.append(obj.value)
            ids.append(obj.id)

        self._values = np.array(values)
        self._ids = np.array(ids)
        self._neighbor_searcher = self._neighbor_searcher.fit(self._values)

    def _update_neighbors_during_insertion(self, object_inserted):
        neighbors = self._get_neighbors(object_inserted)
        for obj in neighbors:
            obj.neighbors.add(object_inserted)
            object_inserted.neighbors.add(obj)

    def _get_neighbors(self, query_object):
        neighbor_indices = self._neighbor_searcher.radius_neighbors(
            [query_object.value], return_distance=False)[0]

        for ix in neighbor_indices:
            id_ = self._ids[ix]
            neighbor = self.objects[id_]
            yield neighbor

    def delete_object(self, obj):
        obj.count -= 1
        if obj.count == 0:
            del self.objects[obj.id]
            self._update_neighbors_during_deletion(obj)
            self.delete_label_of_deleted_object(obj)

    def _update_neighbors_during_deletion(self, object_deleted):
        effective_neighbors = \
            object_deleted.neighbors.difference(set([object_deleted]))
        for neighbor in effective_neighbors:
            neighbor.neighbors.remove(object_deleted)
