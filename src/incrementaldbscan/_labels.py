from typing import Dict

from src.incrementaldbscan._objects import ObjectId

ClusterId = int


class _Labels:
    def __init__(self):
        self._labels: Dict[ObjectId, ClusterId] = dict()

    def set_label(self, object_, label):
        self._labels[object_.id] = label

    def has_label(self, object_id):
        return object_id in self._labels

    def get_label(self, object_):
        return self._labels[object_.id]

    def get_all_labels(self):
        return self._labels

    def delete_label(self, object_id):
        del self._labels[object_id]
