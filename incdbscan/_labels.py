from collections import defaultdict


ClusterLabel = int

CLUSTER_LABEL_UNCLASSIFIED: ClusterLabel = -2
CLUSTER_LABEL_NOISE: ClusterLabel = -1
CLUSTER_LABEL_FIRST_CLUSTER: ClusterLabel = 0


class LabelHandler:
    def __init__(self):
        self._label_to_objects = defaultdict(set)
        self._object_to_label = {}

    def set_label(self, obj, label):
        previous_label = self._object_to_label[obj]
        self._label_to_objects[previous_label].remove(obj)
        self._label_to_objects[label].add(obj)
        self._object_to_label[obj] = label

    def set_label_of_inserted_object(self, obj):
        self._object_to_label[obj] = CLUSTER_LABEL_UNCLASSIFIED
        self._label_to_objects[CLUSTER_LABEL_UNCLASSIFIED].add(obj)

    def set_labels(self, objects, label):
        for obj in objects:
            self.set_label(obj, label)

    def delete_label_of_deleted_object(self, obj):
        label = self.get_label(obj)
        self._label_to_objects[label].remove(obj)

    def get_label(self, obj):
        return self._object_to_label[obj]

    def get_next_cluster_label(self):
        return max(self._label_to_objects.keys()) + 1

    def change_labels(self, change_from, change_to):
        affected_objects = self._label_to_objects.pop(change_from)
        self._label_to_objects[change_to].update(affected_objects)

        for obj in affected_objects:
            self._object_to_label[obj] = change_to
