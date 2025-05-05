from collections import defaultdict

import rustworkx as rx

from ._bfscomponentfinder import BFSComponentFinder
from ._labels import CLUSTER_LABEL_NOISE


class Deleter:
    def __init__(self, eps, min_pts, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.objects = objects

    def delete(self, object_to_delete):
        self.objects.delete_object(object_to_delete)
        object_deleted = object_to_delete

        ex_cores = self._get_objects_that_lost_core_property(object_deleted)

        update_seeds, non_core_neighbors_of_ex_cores = \
            self._get_update_seeds_and_non_core_neighbors_of_ex_cores(
                ex_cores, object_deleted)

        if update_seeds:
            # Only for update seeds belonging to the same cluster do we
            # have to consider if split is needed.

            update_seeds_by_cluster = \
                self._group_objects_by_cluster(update_seeds)

            for seeds in update_seeds_by_cluster.values():
                components = self._find_components_to_split_away(seeds)
                for component in components:
                    self.objects.set_labels(
                        component, self.objects.get_next_cluster_label())

        # Updating labels of border objects that were in the neighborhood
        # of objects that lost their core property is always needed. They
        # become either borders of other clusters or noise.

        self._set_each_border_object_labels_to_largest_around(
            non_core_neighbors_of_ex_cores)

    def _get_objects_that_lost_core_property(self, object_deleted):
        for obj in object_deleted.neighbors:
            if obj.neighbor_count == self.min_pts - 1:
                yield obj

        # The result has to contain the deleted object if it was core
        if object_deleted.is_core:
            yield object_deleted

    def _get_update_seeds_and_non_core_neighbors_of_ex_cores(
            self,
            ex_cores,
            object_deleted):

        update_seeds = set()
        non_core_neighbors_of_ex_cores = set()

        for ex_core in ex_cores:
            # The is-core property of objects that became non core need to be
            # re-cached
            ex_core._clear_is_core_cache()
            for neighbor in ex_core.neighbors:
                if neighbor.is_core:
                    update_seeds.add(neighbor)
                else:
                    non_core_neighbors_of_ex_cores.add(neighbor)

        if object_deleted.count == 0:
            update_seeds = update_seeds.difference({object_deleted})
            non_core_neighbors_of_ex_cores = \
                non_core_neighbors_of_ex_cores.difference({object_deleted})

        return update_seeds, non_core_neighbors_of_ex_cores

    def _group_objects_by_cluster(self, objects):
        grouped_objects = defaultdict(list)

        for obj in objects:
            label = self.objects.get_label(obj)
            grouped_objects[label].append(obj)

        return grouped_objects

    def _find_components_to_split_away(self, seed_objects):
        if len(seed_objects) == 1:
            return []

        if self._objects_are_neighbors_of_each_other(seed_objects):
            return []

        seed_node_ids = [obj.node_id for obj in seed_objects]
        finder = BFSComponentFinder(self.objects.graph)
        rx.bfs_search(self.objects.graph, seed_node_ids, finder)

        seed_of_largest, size_of_largest = 0, 0
        for seed_id, component in finder.seed_to_component.items():
            component_size = len(component)
            if component_size > size_of_largest:
                size_of_largest = component_size
                seed_of_largest = seed_id

        for seed_id, component in finder.seed_to_component.items():
            if seed_id != seed_of_largest:
                yield component

    @staticmethod
    def _objects_are_neighbors_of_each_other(objects):
        for obj1 in objects:
            for obj2 in objects:
                if obj2 not in obj1.neighbors:
                    return False
        return True

    def _set_each_border_object_labels_to_largest_around(self, objects_to_set):
        cluster_updates = {}

        for obj in objects_to_set:
            labels = self._get_cluster_labels_in_neighborhood(obj)
            if not labels:
                labels.add(CLUSTER_LABEL_NOISE)

            cluster_updates[obj] = max(labels)

        for obj, new_cluster_label in cluster_updates.items():
            self.objects.set_label(obj, new_cluster_label)

    def _get_cluster_labels_in_neighborhood(self, obj):
        return {self.objects.get_label(neighbor)
                for neighbor in obj.neighbors
                if neighbor.is_core}
