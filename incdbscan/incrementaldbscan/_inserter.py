import networkx as nx

from ._object import CLUSTER_LABEL_NOISE, CLUSTER_LABEL_UNCLASSIFIED


class _Inserter:
    def __init__(self, eps, min_pts, object_set):
        self.eps = eps
        self.min_pts = min_pts
        self.object_set = object_set

    def insert(self, object_value):
        print('\nInserting', object_value)  # TODO
        object_inserted = self.object_set.insert_object(object_value)

        neighbors = self.object_set.get_neighbors(object_inserted, self.eps)
        self.object_set.update_neighbor_counts_after_insertion(
            object_inserted, neighbors)

        new_core_neighbors, old_core_neighbors = \
            self._filter_core_objects_split_by_novelty(neighbors)

        if not new_core_neighbors:
            print('\nnot new_core_neighbors')  # TODO
            # If there is no new core object, only the new object has to be
            # put in a cluster.

            if old_core_neighbors:
                print('old_core_neighbors')
                # If there are already core objects near to the new object,
                # the new object is put in the most recent cluster. This is
                # similar to case "Absorption" in the paper but not defined
                # there.

                label_of_new_object = max(
                    [obj.label for obj in old_core_neighbors])

            else:
                print('not old_core_neighbors')
                # If the new object does not have any core neighbors,
                # it becomes a noise. Called case "Noise" in the paper.

                label_of_new_object = CLUSTER_LABEL_NOISE

            object_inserted.label = label_of_new_object
            return

        print('\nnew_core_neighbors')  # TODO
        neighbors_of_new_core_neighbors = \
            self.object_set.get_neighbors_of_objects(
                new_core_neighbors, self.eps)

        update_seeds = self._get_update_seeds(neighbors_of_new_core_neighbors)

        connected_components_in_update_seeds = \
            self._get_connected_components(update_seeds)

        for component in connected_components_in_update_seeds:
            effective_cluster_labels = \
                self._get_effective_cluster_labels_of_objects(component)

            if not effective_cluster_labels:
                print('not effective_cluster_labels')  # TODO
                # If in a connected component of update seeds there are only
                # previously unclassified and noise objects, a new cluster is
                # created. Corresponds to case "Creation" in the paper.

                next_cluster_label = self.object_set.get_next_cluster_label()
                self.object_set.set_labels(component, next_cluster_label)

            else:
                print('real_cluster_labels')  # TODO
                # If in a connected component of updates seeds there are
                # already clustered objects, all objects in the component
                # will be merged into the most recent cluster.
                # Corresponds to cases "Absorption" and "Merge" in the paper.

                max_label = max(effective_cluster_labels)
                self.object_set.set_labels(component, max_label)

                for label in effective_cluster_labels:
                    self.object_set.change_labels(label, max_label)

        # Finally all neighbors of each new core object inherits a label from
        # its new core neighbor, thereby affecting border and noise objects,
        # and the object being inserted.

        self._set_cluster_label_to_that_of_new_core_neighbor(
            neighbors_of_new_core_neighbors
        )
        return

    def _filter_core_objects_split_by_novelty(self, objects):
        new_cores = set()
        old_cores = set()

        for obj in objects:
            if obj.neighbor_count == self.min_pts:
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                old_cores.add(obj)

        return new_cores, old_cores

    def _get_update_seeds(self, neighbors_of_new_core_neighbors):
        seeds = set()

        for neighbors in neighbors_of_new_core_neighbors.values():
            core_neighbors = [obj for obj in neighbors
                              if obj.neighbor_count >= self.min_pts]
            seeds.update(core_neighbors)

        return seeds

    def _get_connected_components(self, objects):
        if len(objects) == 1:
            return [objects]

        G = nx.Graph()

        for obj in objects:
            neighbors = self.object_set.get_neighbors(obj, self.eps)

            for neighbor in neighbors:
                if neighbor in objects:
                    G.add_edge(obj, neighbor)

        return nx.connected_components(G)

    def _get_effective_cluster_labels_of_objects(self, objects):
        non_effective_cluster_labels = set([
            CLUSTER_LABEL_UNCLASSIFIED,
            CLUSTER_LABEL_NOISE
        ])
        effective_cluster_labels = set()

        for obj in objects:
            label = obj.label
            if label not in non_effective_cluster_labels:
                effective_cluster_labels.add(label)

        return effective_cluster_labels

    def _set_cluster_label_to_that_of_new_core_neighbor(
            self,
            neighbors_of_new_core_neighbors):

        for new_core, neighbors in neighbors_of_new_core_neighbors.items():
            self.object_set.set_labels(neighbors, new_core.label)
