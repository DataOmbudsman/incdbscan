from collections import defaultdict

import networkx as nx

from incdbscan.incrementaldbscan._labels import CLUSTER_LABEL_NOISE


class _Deleter:
    def __init__(self, eps, min_pts, labels, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = labels
        self.objects = objects

    def delete(self, object_id):
        print('\nDeleting', object_id)
        object_to_delete = self.objects.get_object(object_id)
        self.objects.delete_object(object_id)

        neighbors = self.objects.get_neighbors(object_to_delete, self.eps)
        self._update_neighbor_counts_after_deletion(neighbors)

        ex_cores = self._get_objects_that_lost_core_property(
            neighbors, object_to_delete)

        neighbors_of_ex_cores = \
            self.objects.get_neighbors_of_objects(ex_cores, self.eps)

        update_seeds, non_core_neighbors_of_ex_cores = \
            self._separate_neighbors_by_core_property(neighbors_of_ex_cores)

        if update_seeds:
            print('\nupdate_seeds', update_seeds)  # TODO

            # Only for update seeds belonging to the same cluster do we
            # have to consider if split is needed.

            update_seeds_by_cluster = \
                self._group_objects_by_cluster(update_seeds)

            for seeds in update_seeds_by_cluster.values():
                components = self._find_components_to_split_away(
                    seeds, non_core_neighbors_of_ex_cores)

                for component in components:
                    self.labels.set_labels(
                        component, self.labels.get_next_cluster_label())

        # Updating labels of border objects that were in the neighborhood
        # of objects that lost their core property is always needed. They
        # become either borders of other clusters or noise.

        self._set_each_border_object_labels_to_largest_around(
            non_core_neighbors_of_ex_cores)
        self.labels.delete_label(object_id)

    def _update_neighbor_counts_after_deletion(self, neighbors):
        for neighbor in neighbors:
            neighbor.neighbor_count -= 1

    def _get_objects_that_lost_core_property(
            self,
            neighbors,
            object_to_delete):

        ex_core_neighbors = [
            obj for obj in neighbors
            if obj.neighbor_count == self.min_pts - 1
        ]

        # The result has to contain the deleted object if it was core

        if self._is_core(object_to_delete):
            ex_core_neighbors.append(object_to_delete)

        return ex_core_neighbors

    def _is_core(self, obj):
        return obj.neighbor_count >= self.min_pts

    def _separate_neighbors_by_core_property(self, neighbors_of_ex_cores):
        core_objects = set()
        non_core_objects = set()

        for neighbors in neighbors_of_ex_cores.values():
            for neighbor in neighbors:
                if self._is_core(neighbor):
                    core_objects.add(neighbor)
                else:
                    non_core_objects.add(neighbor)

        return core_objects, non_core_objects

    def _group_objects_by_cluster(self, objects):
        objects_with_cluster_labels = [
            (obj, self.labels.get_label(obj)) for obj in objects
        ]
        grouped_objects = defaultdict(list)

        for obj, label in objects_with_cluster_labels:
            grouped_objects[label].append(obj)

        return grouped_objects

    def _find_components_to_split_away(
            self,
            seed_objects,
            objects_to_exclude_from_components):

        # Traverse the objects in a BFS manner to find those components of
        # objects that need to be split away. Starting from the seed objects,
        # expand the graph by adding neighboring objects, but objects whose
        # update is not taken care of in this step are excluded. Traversal
        # terminates when all of the next nodes to be added are linked to the
        # same seed objects -- this means that the components (that all can be
        # linked to other seed objects) are traversed completely and they can
        # be split away.

        if len(seed_objects) == 1:
            return list()

        if self._objects_are_neighbors_of_each_other(seed_objects):
            return list()

        def _expand_graph(obj, seed_id):
            nodes = set(G.nodes)
            neighbors = self.objects.get_neighbors(obj, self.eps)

            for neighbor in neighbors:
                if neighbor not in objects_to_exclude_from_components:
                    G.add_edge(obj, neighbor)
                if neighbor not in nodes and self._is_core(neighbor):
                    nodes_to_visit.append((neighbor, seed_id))

        def _nodes_to_visit_are_from_different_seeds():
            seed_ids = set([seed_id for (node, seed_id) in nodes_to_visit])
            return len(seed_ids) > 1

        G = nx.Graph()
        nodes_to_visit = [(seed, seed.id) for seed in seed_objects]

        while _nodes_to_visit_are_from_different_seeds():
            obj, seed_ix = nodes_to_visit.pop(0)
            _expand_graph(obj, seed_ix)

        connected_components = nx.connected_components(G)
        remaining_seed_id = nodes_to_visit[0][1]

        for component in connected_components:
            if all([remaining_seed_id != obj.id for obj in component]):
                yield component

    def _objects_are_neighbors_of_each_other(self, objects):
        for obj1 in objects:
            neighbors = self.objects.get_neighbors(obj1, self.eps)
            for obj2 in objects:
                if obj2 not in neighbors:
                    return False
        return True

    def _set_each_border_object_labels_to_largest_around(
            self,
            objects_to_set):

        cluster_updates = {}

        for obj in objects_to_set:
            labels = self._get_cluster_labels_in_neighborhood(obj)
            if not labels:
                labels.add(CLUSTER_LABEL_NOISE)

            cluster_updates[obj] = max(labels)

        for obj, new_cluster_label in cluster_updates.items():
            self.labels.set_label(obj, new_cluster_label)

    def _get_cluster_labels_in_neighborhood(self, obj):
        labels = set()

        for neighbor in self.objects.get_neighbors(obj, self.eps):
            if self._is_core(neighbor):
                label_of_neighbor = self.labels.get_label(neighbor)
                labels.add(label_of_neighbor)

        return labels
