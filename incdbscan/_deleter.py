from collections import (
    defaultdict,
    deque
)
from functools import lru_cache
from typing import (
    Dict,
    List
)

import rustworkx as rx
from rustworkx.visit import (
    BFSVisitor,
    PruneSearch
)

from ._labels import CLUSTER_LABEL_NOISE
from ._object import (
    NodeId,
    Object
)


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
        grouped_objects = defaultdict(list)

        for obj in objects:
            label = self.objects.get_label(obj)
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

        finder = _ComponentFinder(self.objects.graph, self._is_core)
        seed_node_ids = [obj.node_id for obj in seed_objects]
        rx.bfs_search(self.objects.graph, seed_node_ids, finder)

        seed_of_largest, size_of_largest = 0, 0
        for seed_id, objects in finder.seed_to_objects.items():
            component_size = len(objects)
            if component_size > size_of_largest:
                size_of_largest = component_size
                seed_of_largest = seed_id

        for seed_id, objects in finder.seed_to_objects.items():
            if seed_id != seed_of_largest:
                yield objects

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


class _ComponentFinder(BFSVisitor):

    def __init__(self, graph, is_core_fn):
        self.graph = graph
        self.is_core = is_core_fn
        self.seed_to_objects: Dict[NodeId, List[Object]] = defaultdict(set)
        self.node_to_seed: Dict[NodeId, NodeId] = defaultdict(int)

    def discover_vertex(self, vertex_node_id):
        # If this is the first time discovering a node then the node itself
        # will be its own seed. This is the way we keep track of singleton
        # nodes (i.e., ones without edges).

        if vertex_node_id not in self.node_to_seed:
            self.node_to_seed[vertex_node_id] = vertex_node_id
            self.seed_to_objects[vertex_node_id].add(self.graph[vertex_node_id])

        # If the node does not represent a core object then we don't want
        # traversal to go in that direction.

        if not self.is_core(self.graph[vertex_node_id]):
            raise PruneSearch

    def tree_edge(self, edge):
        source_node_id, target_node_id, _ = edge

        # The target of the edge is a new node we see for the first time. Its
        # seed will be the seed of the source.

        self.node_to_seed[target_node_id] = self.node_to_seed[source_node_id]
        seed = self.node_to_seed[target_node_id]
        self.seed_to_objects[seed].add(self.graph[target_node_id])

    def non_tree_edge(self, edge):
        source_node_id, target_node_id, _ = edge

        # A non-tree edge is the case of merge, that is, when two components
        # with different seeds meet. However, we only merge them if the target
        # represents core object in the graph (i.e., dense connection).

        source_seed = self.node_to_seed[source_node_id]
        target_seed = self.node_to_seed[target_node_id]
        different_seeds = source_seed != target_seed

        if different_seeds and self.is_core(self.graph[target_node_id]):
            if source_seed > target_seed:
                self.node_to_seed[target_node_id] = source_seed
            else:
                self.node_to_seed[source_node_id] = target_seed

        seed = self.node_to_seed[target_node_id]
        self.seed_to_objects[seed].add(self.graph[target_node_id])
