from typing import Dict, Iterable

import networkx as nx

from src.incrementaldbscan._labels import ClusterLabel
from src.incrementaldbscan._objects import _Object


class _Updater():
    def __init__(self, incdbscan):
        self.incdbscan = incdbscan
        self.eps = incdbscan.eps
        self.min_pts = incdbscan.min_pts
        self.labels = incdbscan._labels
        self.objects = incdbscan._objects

    def insertion(self, object_to_insert: _Object):
        print('\nInserting', object_to_insert.id)  # TODO
        self.objects.add_object(object_to_insert)
        self.labels.set_label(
            object_to_insert, self.incdbscan.CLUSTER_LABEL_UNCLASSIFIED)

        neighbors = self.objects.get_neighbors(object_to_insert, self.eps)
        self._update_neighbor_counts_after_insert(object_to_insert, neighbors)

        new_core_neighbors, old_core_neighbors = \
            self._filter_core_objects_split_by_novelty(neighbors)

        if not new_core_neighbors:
            print('\nnot new_core_neighbors')  # TODO
            # If there is no new core object,
            # only the new object has to be put in a cluster.

            if old_core_neighbors:
                print('old_core_neighbors')
                # If there are already core objects near to the new object,
                # the new object is put in the most recent cluster. This is
                # similar to case "Absorption" in the paper but not defined
                # there.

                label_of_new_object = max([
                    self.labels.get_label(obj) for obj in old_core_neighbors
                ])

            else:
                print('not old_core_neighbors')
                # If the new object does not have any core neighbors,
                # it becomes a noise. Called case "Noise" in the paper.

                label_of_new_object = self.incdbscan.CLUSTER_LABEL_NOISE

            self.labels.set_label(
                    object_to_insert, label_of_new_object)
            return

        print('\nnew_core_neighbors')  # TODO
        neighbors_of_new_core_neighbors = \
            self._get_neighbors_of_objects(new_core_neighbors)

        update_seeds = self._get_update_seeds(neighbors_of_new_core_neighbors)

        connected_components_in_update_seeds = \
            self._get_connected_components_in_update_seeds(
                update_seeds, neighbors_of_new_core_neighbors)

        for component in connected_components_in_update_seeds:
            real_cluster_labels = \
                self._get_real_cluster_labels_of_objects(component)

            if not real_cluster_labels:
                print('not real_cluster_labels')  # TODO
                # If in a connected component of update seeds there are only
                # previously unclassified and noise objects, a new cluster is
                # created. Corresponds to case "Creation" in the paper.

                for obj in component:
                    self.labels.set_label(
                        obj, self.incdbscan._next_cluster_label)
                self.incdbscan._next_cluster_label += 1

            else:
                print('real_cluster_labels')  # TODO
                # If in a connected component of updates seeds there are
                # already clustered objects, all objects in the component
                # will be merged into the most recent cluster.
                # Corresponds to cases "Absorption" and "Merge" in the paper.

                max_label = max(real_cluster_labels)

                for obj in component:
                    self.labels.set_label(obj, max_label)

                for label in real_cluster_labels:
                    self.labels.change_labels(label, max_label)

        # Finally all neighbors of each new core object inherits a label from
        # its new core neighbor, thereby covering border and noise objects.
        self._set_cluster_label_to_that_of_new_core_neighbor(
            neighbors_of_new_core_neighbors
        )

    def _update_neighbor_counts_after_insert(self, new_object, neighbors):
        for neighbor in neighbors:
            neighbor.neighbor_count += 1
        new_object.neighbor_count = len(neighbors)

    def _filter_core_objects_split_by_novelty(self, objects):
        new_cores = set()
        old_cores = set()

        for obj in objects:
            if obj.neighbor_count == self.min_pts:
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                old_cores.add(obj)

        return new_cores, old_cores

    def _get_neighbors_of_objects(self, objects):
        neighbors = dict()

        for obj in objects:
            neighbors[obj] = \
                self.objects.get_neighbors(obj, self.eps)

        return neighbors

    def _get_update_seeds(self, neighbors_dict):
        """
        During insertion, neighbors_dict holds neighbors of new core neighbors.
        During deletion, neighbors_dict hold neighbors of ex core neighbors.
        """
        seeds = set()

        for neighbors in neighbors_dict.values():
            core_neighbors = [obj for obj in neighbors
                              if obj.neighbor_count >= self.min_pts]
            seeds.update(core_neighbors)

        return seeds

    def _get_connected_components_in_update_seeds(
            self,
            update_seeds: Iterable,
            stored_neighbors: Dict):

        G = nx.Graph()

        for seed in update_seeds:

            if seed in stored_neighbors:
                neighbors = stored_neighbors[seed]
            else:
                neighbors = \
                    self.objects.get_neighbors(seed, self.eps)

            for neighbor in neighbors:
                if neighbor.neighbor_count >= self.min_pts:
                    G.add_edge(seed, neighbor)

        return nx.connected_components(G)

    def _get_real_cluster_labels_of_objects(self, objects):
        real_cluster_labels = set()

        for obj in objects:
            label = self.labels.get_label(obj)
            if label not in self.incdbscan.TECHNICAL_CLUSTER_LABELS:
                real_cluster_labels.add(label)

        return real_cluster_labels

    def _set_cluster_label_to_that_of_new_core_neighbor(
            self,
            neighbors_of_new_core_neighbors):

        for new_core, neighbors in neighbors_of_new_core_neighbors.items():
            label = self.labels.get_label(new_core)
            for neighbor in neighbors:
                self.labels.set_label(neighbor, label)

    def deletion(self, object_id):
        print('\nDeleting', object_id)
        object_to_delete = self.objects.get_object(object_id)
        self.objects.remove_object(object_id)

        neighbors = self.objects.get_neighbors(object_to_delete, self.eps)
        self._update_neighbor_counts_after_deletion(neighbors)

        ex_core_neighbors = [
            obj for obj in neighbors if obj.neighbor_count == self.min_pts - 1
        ]

        if object_to_delete.neighbor_count >= self.min_pts:
            ex_core_neighbors.append(object_to_delete)

        neighbors_of_ex_core_neighbors = \
            self._get_neighbors_of_objects(ex_core_neighbors)

        update_seeds = self._get_update_seeds(neighbors_of_ex_core_neighbors)

        # TODO what to refactor from above?

        if not update_seeds:
            print('not update_seeds')  # TODO
            # If there are no update seeds, only border objects might change
            # their cluster assignment, either to noise or to the cluster id of
            # a core object in their neighborhood. Similar to, but a fixed
            # version of case "Removal" in the paper

            for ex_core in neighbors_of_ex_core_neighbors.keys():
                label_of_ex_core = self.labels.get_label(ex_core)
                neighbors_of_ex_core = neighbors_of_ex_core_neighbors[ex_core]

                self._set_object_labels_to_largest_around_in_parallel(
                    objects_to_set=neighbors_of_ex_core,
                    excluded_labels=[label_of_ex_core]
                )

            self.labels.delete_label(object_id)
            return

        print('update seeds')

        # TODO Talán a végére ez kell? self.labels.delete_label(object_id)

    def _update_neighbor_counts_after_deletion(self, neighbors):
        for neighbor in neighbors:
            neighbor.neighbor_count -= 1

    def _set_object_labels_to_largest_around_in_parallel(
            self,
            objects_to_set,
            excluded_labels):

        cluster_updates = {}

        for obj in objects_to_set:
            labels = self._get_cluster_labels_in_neighborhood(obj)
            labels.difference_update(excluded_labels)
            if not labels:
                labels.add(self.incdbscan.CLUSTER_LABEL_NOISE)

            cluster_updates[obj] = max(labels)

        for obj, new_cluster_label in cluster_updates.items():
            self.labels.set_label(obj, new_cluster_label)

    def _get_cluster_labels_in_neighborhood(self, obj):
        labels = set()

        for neighbor in self.objects.get_neighbors(obj, self.eps):
            label_of_neighbor = self.labels.get_label(neighbor)
            labels.add(label_of_neighbor)

        return labels
