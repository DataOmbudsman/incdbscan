from typing import Dict

from incdbscan.incrementaldbscan._objects import ObjectId

ClusterLabel = int

CLUSTER_LABEL_UNCLASSIFIED: ClusterLabel = -2
CLUSTER_LABEL_NOISE: ClusterLabel = -1
CLUSTER_LABEL_FIRST_CLUSTER: ClusterLabel = 0


class _Labels:
    def __init__(self):
        self._labels: Dict[ObjectId, ClusterLabel] = dict()

    def set_label(self, obj, label):
        self._labels[obj.id] = label

    def set_labels(self, objects, label):
        for obj in objects:
            self.set_label(obj, label)

    def has_label(self, object_id):
        return object_id in self._labels

    def get_label(self, obj):
        return self._labels[obj.id]

    def get_all_labels(self):
        return self._labels

    def delete_label(self, object_id):
        del self._labels[object_id]

    def change_labels(self, change_from, change_to):
        for object_id, cluster_label in self._labels.items():
            if cluster_label == change_from:
                self._labels[object_id] = change_to

    def get_next_cluster_label(self):
        if not self._labels:
            return CLUSTER_LABEL_FIRST_CLUSTER
        return max(self._labels.values()) + 1
