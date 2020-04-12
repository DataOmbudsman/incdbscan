from collections import defaultdict

import networkx as nx

from src.incrementaldbscan._labels import CLUSTER_LABEL_NOISE


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
            neighbors,
            object_to_delete
        )

        neighbors_of_ex_cores = \
            self.objects.get_neighbors_of_objects(ex_cores, self.eps)

        update_seeds = self._get_update_seeds(neighbors_of_ex_cores)

        if update_seeds:
            print('\nupdate_seeds', update_seeds)  # TODO

            # Only for update seeds belonging to the same cluster do we
            # have to consider if split is needed.

            update_seeds_by_cluster = \
                self._group_objects_by_cluster(update_seeds)

            for seeds in update_seeds_by_cluster.values():
                components = self._find_components_to_split_away(seeds)
                # print(list(components)) #TODO

                for component in components:
                    self.labels.set_labels(
                        component, self.labels.get_next_cluster_label())

        # Updating labels of border objects that were in the neighborhood
        # of objects that lost their core property is always needed. They
        # become either borders of other clusters or noise.

        self._update_labels_of_border_objects_around_ex_cores(
            neighbors_of_ex_cores)
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
        if object_to_delete.neighbor_count >= self.min_pts:
            ex_core_neighbors.append(object_to_delete)

        return ex_core_neighbors

    def _get_update_seeds(self, neighbors_of_ex_cores):
        seeds = set()

        for neighbors in neighbors_of_ex_cores.values():
            core_neighbors = [obj for obj in neighbors
                              if obj.neighbor_count >= self.min_pts]
            seeds.update(core_neighbors)

        return seeds

    def _update_labels_of_border_objects_around_ex_cores(
            self,
            neighbors_of_ex_cores):

        for neighbors_of_ex_core in neighbors_of_ex_cores.values():
            self._set_border_object_labels_to_largest_around_in_parallel(
                objects_to_set=neighbors_of_ex_core,
            )

    def _set_border_object_labels_to_largest_around_in_parallel(
            self,
            objects_to_set):

        cluster_updates = {}

        for obj in objects_to_set:
            if obj.neighbor_count < self.min_pts:
                labels = self._get_cluster_labels_in_neighborhood(obj)
                if not labels:
                    labels.add(CLUSTER_LABEL_NOISE)

                cluster_updates[obj] = max(labels)

        for obj, new_cluster_label in cluster_updates.items():
            self.labels.set_label(obj, new_cluster_label)

    def _get_cluster_labels_in_neighborhood(self, obj):
        labels = set()

        for neighbor in self.objects.get_neighbors(obj, self.eps):
            if neighbor.neighbor_count >= self.min_pts:
                label_of_neighbor = self.labels.get_label(neighbor)
                labels.add(label_of_neighbor)

        return labels

    def _find_components_to_split_away(self, seed_objects):
        if len(seed_objects) == 1:
            return list()

        if self._objects_are_neighbors_of_each_other(seed_objects):
            return list()

        G = nx.Graph()
        nodes_to_visit = list()

        print("HEREr")

        def _add_neighbors_of_object_to_graph(obj, seed_id):
            edges = set(G.edges)
            nodes = set(G.nodes)

            neighbors = self.objects.get_neighbors(obj, self.eps)

            for neighbor in neighbors:
                if neighbor != obj:
                    if (obj, neighbor) not in edges:
                        G.add_edge(obj, neighbor)
                    if neighbor not in nodes:
                        nodes_to_visit.append((neighbor, seed_id))

        def _nodes_to_visit_are_from_different_seeds():
            seed_ids = set([seed_id for (node, seed_id) in nodes_to_visit])
            return len(seed_ids) > 1

        for seed in seed_objects:
            _add_neighbors_of_object_to_graph(seed, seed.id)

        print('0 print', nodes_to_visit); print()
        while _nodes_to_visit_are_from_different_seeds():
            obj, seed_ix = nodes_to_visit.pop(0)
            _add_neighbors_of_object_to_graph(obj, seed_ix)
            print('END print', nodes_to_visit)

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

    def _group_objects_by_cluster(self, objects):
        objects_with_cluster_labels = [
            (obj, self.labels.get_label(obj)) for obj in objects
        ]
        grouped_objects = defaultdict(list)

        for obj, label in objects_with_cluster_labels:
            grouped_objects[label].append(obj)

        return grouped_objects
