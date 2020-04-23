from typing import Dict, List

import numpy as np

from ._utils import hash_

ObjectId = str


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


class _Object:
    def __init__(self, value, id_):
        self.value = value
        self.id = id_
        self.count = 1
        self.neighbor_count = 0


class _ObjectSet:
    def __init__(self, distance=euclidean_distance):
        self.objects = dict()
        self.distance = distance

    def get_object(self, value):
        id_ = hash_(value)
        return self.objects.get(id_)

    def get_neighbors(self, query_object, radius):
        return [
            obj for obj in self.objects.values()
            if self.distance(query_object.value, obj.value) <= radius
        ]

    def get_neighbors_of_objects(
            self, query_objects, radius) -> Dict[ObjectId, List[ObjectId]]:

        neighbors = dict()

        for obj in query_objects:
            neighbors[obj] = self.get_neighbors(obj, radius)

        return neighbors

    def insert_object(self, value):
        id_ = hash_(value)

        if id_ in self.objects:
            obj = self.objects[id_]
            obj.count += 1

        else:
            obj = _Object(value, id_)
            self.objects[id_] = obj

        return obj

    @staticmethod
    def update_neighbor_counts_after_insertion(
            inserted,
            neighbors_of_inserted):

        total_count = 0

        for neighbor in neighbors_of_inserted:
            neighbor.neighbor_count += 1
            total_count += neighbor.count

        # rewrite neighbor_count of inserted object
        inserted.neighbor_count = total_count

    def delete_object(self, obj):
        obj.count -= 1
        if obj.count == 0:
            del self.objects[obj.id]

    @staticmethod
    def update_neighbor_counts_after_deletion(neighbors):
        for neighbor in neighbors:
            neighbor.neighbor_count -= 1
