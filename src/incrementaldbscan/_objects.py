import numpy as np
from typing import Dict, Union

ObjectId = Union[int, str]


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


class _Object:
    def __init__(self, object_value, object_id):
        self.value = object_value
        self.id = object_id
        self.neighbor_count = 1


class _Objects:
    def __init__(self, distance=euclidean_distance):
        self.objects: Dict[ObjectId, _Object] = dict()
        self._distance = distance

    def add_object(self, object_to_add: _Object):
        self.objects[object_to_add.id] = object_to_add

    def remove_object(self, object_id: ObjectId):
        del self.objects[object_id]

    def get_all(self):
        return self.objects.values()

    def get_neighbors(
            self,
            query_object,
            radius):

        return [
            object_ for object_ in self.get_all()
            if self._distance(query_object.value, object_.value) <= radius
        ]