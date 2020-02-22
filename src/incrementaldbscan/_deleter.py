from src.incrementaldbscan._labels import CLUSTER_LABEL_NOISE


class _Deleter():
    def __init__(self, eps, min_pts, labels, objects):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = labels
        self.objects = objects

    def delete(self, object_id):
        print('\nDeleting', object_id)
        object_to_delete = self.objects.get_object(object_id)
        self.objects.remove_object(object_id)

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
            print('\nupdate_seeds')  # TODO
            n_components = len(
                    self._get_connected_components_by_expansion(update_seeds)
            )

            if n_components > 1:
                pass
                # Splitting logic

        # Updating labels of border objects that were in the neighborhood
        # of objects that lost their core property is always needed. They
        # become either borders of other clusters or noise.

        self._update_labels_of_border_objects_of_ex_cores(
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

    def _update_labels_of_border_objects_of_ex_cores(
            self,
            neighbors_of_ex_cores):

        for ex_core, neighbors_of_ex_core in neighbors_of_ex_cores.items():
            label_of_ex_core = self.labels.get_label(ex_core)

            self._set_border_object_labels_to_largest_around_in_parallel(
                objects_to_set=neighbors_of_ex_core,
                excluded_labels=[label_of_ex_core]
            )

    def _set_border_object_labels_to_largest_around_in_parallel(
            self,
            objects_to_set,
            excluded_labels):

        cluster_updates = {}

        for obj in objects_to_set:
            if obj.neighbor_count < self.min_pts:
                labels = self._get_cluster_labels_in_neighborhood(obj)
                labels.difference_update(excluded_labels)
                if not labels:
                    labels.add(CLUSTER_LABEL_NOISE)

                cluster_updates[obj] = max(labels)

        for obj, new_cluster_label in cluster_updates.items():
            self.labels.set_label(obj, new_cluster_label)

    def _get_cluster_labels_in_neighborhood(self, obj):
        labels = set()

        for neighbor in self.objects.get_neighbors(obj, self.eps):
            label_of_neighbor = self.labels.get_label(neighbor)
            labels.add(label_of_neighbor)

        return labels

    def _get_connected_components_by_expansion(self, objects):
        return list(['dummy'])
