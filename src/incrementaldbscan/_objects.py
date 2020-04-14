from functools import lru_cache
from typing import Dict, List, Union

import numpy as np

ObjectId = Union[int, str]


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


class _Object:
    def __init__(self, object_value, object_id):
        self.value = object_value
        self.id = object_id
        self.neighbor_count = 1

    def __repr__(self):
        return f'<ID: {self.id}, count: {self.neighbor_count}>'


class _Objects:
    def __init__(self, cache_size, distance=euclidean_distance):
        self.objects: Dict[ObjectId, _Object] = dict()
        self._distance = distance
        self.get_neighbors = lru_cache(maxsize=cache_size)(self.get_neighbors)

    def get_object(self, object_id: ObjectId):
        return self.objects[object_id]

    def get_neighbors(self, query_object, radius):
        return [
            object_ for object_ in self.objects.values()
            if self._distance(query_object.value, object_.value) <= radius
        ]

    def get_neighbors_of_objects(
            self, query_objects, radius) -> Dict[ObjectId, List[ObjectId]]:

        neighbors = dict()

        for obj in query_objects:
            neighbors[obj] = self.get_neighbors(obj, radius)

        return neighbors

    def add_object(self, object_to_add: _Object):
        self.objects[object_to_add.id] = object_to_add
        self.get_neighbors.cache_clear()

    def delete_object(self, object_id: ObjectId):
        del self.objects[object_id]
        self.get_neighbors.cache_clear()
