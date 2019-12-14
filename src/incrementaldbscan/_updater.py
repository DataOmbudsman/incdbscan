from typing import Dict, Iterable

import networkx as nx

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
                # akin to case "Absorption" in the paper but not defined
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
            max_cluster_label = max([
                self.labels.get_label(obj) for obj in component
            ])

            if max_cluster_label < 0:
                print('max_cluster_label < 0')  # TODO
                # If in a connected component of update seeds there are only
                # unclassified and noise objects, a new cluster is created.
                # Corresponds to case "Creation" in the paper.

                for obj in component:
                    self.labels.set_label(
                        obj, self.incdbscan._next_cluster_label)
                self.incdbscan._next_cluster_label += 1

            else:
                print('max_cluster_label >= 0')  # TODO
                # If in a connected component of updates seeds there are
                # already clustered objects, all objects in the component
                # will be merged into the most recent cluster.
                # Corresponds to cases "Absorption" and "Merge" in the paper.

                for obj in component:
                    self.labels.set_label(obj, max_cluster_label)

                # TODO környezetet is ???
                # Mindet update-eljük vagy mergeöket számon tartjuk

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

    def _get_update_seeds(self, neighbors_of_new_core_neighbors):
        seeds = set()

        for neighbors in neighbors_of_new_core_neighbors.values():
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

    def _set_cluster_label_to_that_of_new_core_neighbor(
            self,
            neighbors_of_new_core_neighbors):

        for new_core, neighbors in neighbors_of_new_core_neighbors.items():
            label = self.labels.get_label(new_core)
            for neighbor in neighbors:
                self.labels.set_label(neighbor, label)

    def deletion(self, object_id):
        # Do lot of stuff, then:
        self.labels.delete_label(object_id)
