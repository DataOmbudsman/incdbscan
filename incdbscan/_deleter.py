from collections import defaultdict
from functools import lru_cache

import networkx as nx

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

        self._is_core.cache_clear()

    def _get_objects_that_lost_core_property(self, object_deleted):
        ex_core_neighbors = [obj for obj in object_deleted.neighbors
                             if obj.neighbor_count == self.min_pts - 1]

        # The result has to contain the deleted object if it was core

        if self._is_core(object_deleted):
            ex_core_neighbors.append(object_deleted)

        return ex_core_neighbors

    @lru_cache(maxsize=None)
    def _is_core(self, obj):
        return obj.neighbor_count >= self.min_pts

    def _get_update_seeds_and_non_core_neighbors_of_ex_cores(
            self,
            ex_cores,
            object_deleted):

        update_seeds = set()
        non_core_neighbors_of_ex_cores = set()

        for ex_core in ex_cores:
            for neighbor in ex_core.neighbors:
                if self._is_core(neighbor):
                    update_seeds.add(neighbor)
                else:
                    non_core_neighbors_of_ex_cores.add(neighbor)

        if object_deleted.count == 0:
            update_seeds = update_seeds.difference({object_deleted})
            non_core_neighbors_of_ex_cores = \
                non_core_neighbors_of_ex_cores.difference({object_deleted})

        return update_seeds, non_core_neighbors_of_ex_cores

    def _group_objects_by_cluster(self, objects):
        objects_with_cluster_labels = [(obj, self.objects.get_label(obj))
                                       for obj in objects]

        grouped_objects = defaultdict(list)

        for obj, label in objects_with_cluster_labels:
            grouped_objects[label].append(obj)

        return grouped_objects

    def _find_components_to_split_away(self, seed_objects):

        # Traverse the objects in a BFS manner to find those components of
        # objects that need to be split away. A component here is a group of
        # objects that all can be linked to the same seed object. Starting from
        # the seed objects, expand the graph by adding neighboring objects.
        # Traversal terminates when all of the next nodes to be added are
        # linked to the same seed object -- this means that all but one
        # component are traversed completely and they can be split away.

        if len(seed_objects) == 1:
            return []

        if self._objects_are_neighbors_of_each_other(seed_objects):
            return []

        # First, initialize graph and node queue

        graph = nx.Graph()
        nodes_to_visit = []

        for seed in seed_objects:
            graph.add_node(seed)
            nodes_to_visit.append((seed, seed.id))

        # Then, traverse graph

        def _nodes_to_visit_are_from_different_seeds():
            seed_ids = {seed_id for (node, seed_id) in nodes_to_visit}
            return len(seed_ids) > 1

        def _expand_graph(obj, seed_id):
            nodes = set(graph.nodes)

            for neighbor in obj.neighbors:
                neighbor_is_core = self._is_core(neighbor)
                neighbor_not_in_nodes = neighbor not in nodes

                if neighbor_is_core or neighbor_not_in_nodes:
                    graph.add_edge(obj, neighbor)
                if neighbor_is_core and neighbor_not_in_nodes:
                    nodes_to_visit.append((neighbor, seed_id))

        while _nodes_to_visit_are_from_different_seeds():
            obj, seed_ix = nodes_to_visit.pop(0)
            _expand_graph(obj, seed_ix)

        # Finally, find components that need to be split away

        connected_components = nx.connected_components(graph)
        remaining_seed_id = nodes_to_visit[0][1]

        for component in connected_components:
            if all(remaining_seed_id != obj.id for obj in component):
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
                if self._is_core(neighbor)}
