import warnings
from typing import Dict, Iterable

import networkx as nx

from src.incrementaldbscan._labels import ClusterId, _Labels
from src.incrementaldbscan._objects import _Object, _Objects, ObjectId
from src.dbscan._dbscanbase import _DBSCANBase


class IncrementalDBSCAN(_DBSCANBase):
    """
    Based (mostly but not completely) on "Incremental Clustering for Mining
    in a Data Warehousing Environment by Ester et al. 1998.
    """

    def __init__(self, eps, min_pts):
        super().__init__(eps, min_pts)
        self._labels = _Labels()
        self._objects = _Objects()
        self.labels: Dict[ObjectId, ClusterId] = self._labels.get_all_labels()

    def add_object(self, object_value, object_id: ObjectId):
        """
        Add an object (given by its value), along with its ID, to clustering.
        ID is of an unhashable type, such as int or str.
        """

        if not self._labels.has_label(object_id):
            object_to_insert = _Object(object_value, object_id)
            self._insertion(object_to_insert)

        else:
            warnings.warn(
                IncrementalDBSCANWarning(
                    f'Object with ID {object_id} was not added '
                    'because it already exists in the clustering.'
                )
            )

    def add_objects(
            self,
            object_values: Iterable,
            object_ids: Iterable[ObjectId]):
        """
        Add objects (given by their values), along with their IDs, to
        clustering. ID is of an unhashable type, such as int or str.
        """

        for object_value, object_id in zip(object_values, object_ids):
            self.add_object(object_value, object_id)

    def remove_object(self, object_id: ObjectId):
        """Remove object, given by its ID, from clustering."""

        if self._labels.has_label(object_id):
            self._deletion(object_id)

        else:
            warnings.warn(
                IncrementalDBSCANWarning(
                    f'Object with ID {object_id} was not removed '
                    f'because there is no object with this ID.'
                )
            )

    def remove_objects(self, object_ids: Iterable):
        """Remove objects, given by their IDs, from clustering."""

        for object_id in object_ids:
            self.remove_object(object_id)

    def _insertion(self, object_to_insert: _Object):
        # print('\nInserting', object_to_insert.id)  # TODO
        self._objects.add_object(object_to_insert)
        self._labels.set_label(
            object_to_insert, self.CLUSTER_LABEL_UNCLASSIFIED)

        neighbors = self._objects.get_neighbors(object_to_insert, self.eps)
        for neighbor in neighbors:
            neighbor.neighbor_count += 1
        object_to_insert.neighbor_count = len(neighbors)

        new_core_neighbors, old_core_neighbors = \
            self._filter_core_objects_split_by_novelty(neighbors)

        if not new_core_neighbors:
            print('\nnot new_core_neighbors')  # TODO
            # If there is no new core object,
            # only the new object has to be put in a cluster.

            if old_core_neighbors:
                print('old_core_neighbors')
                # If there are already core objects near to the new object,
                # the new object is put in the most recent cluster.
                # n.b. This is a case not defined by the paper.
                label_of_new_object = max([
                    self._labels.get_label(obj) for obj in old_core_neighbors
                ])

            else:
                print('not old_core_neighbors')
                # If the new object does not have any core neighbors,
                # it becomes a noise. Called case "Noise" in the paper.
                label_of_new_object = self.CLUSTER_LABEL_NOISE

            self._labels.set_label(
                    object_to_insert, label_of_new_object)
            return

        print('\nnew_core_neighbors')  # TODO
        neighbors_of_new_core_neighbors = \
            self._get_neighbors_of_objects(new_core_neighbors)

        update_seeds = self._get_update_seeds(neighbors_of_new_core_neighbors)

        for component in self._get_connected_components(update_seeds):
            max_cluster_label = max([
                self._labels.get_label(obj) for obj in component
            ])

            if max_cluster_label < 0:
                print('max_cluster_label < 0')  # TODO
                # If in a connected component of update seeds there are only
                # noise objects, a new cluster is created.
                # Similar to case "Creation" in the paper
                for obj in component:
                    self._labels.set_label(obj, self._next_cluster_label)
                self._next_cluster_label += 1

            else:
                print('max_cluster_label >= 0')  # TODO
                # If in a connected component of updates seeds there are
                # already clustered objects, all objects in the component
                # will be merged into the most recent cluster.
                # Similar to case "Absorption" and case "Merge" in the paper
                for obj in component:
                    self._labels.set_label(obj, max_cluster_label)

                # TODO környezetet is ???
                # Mindet update-eljük vagy mergeöket számon tartjuk

        # Finally all neighbors of each new core object will get a label
        # corresponding to the new core to cover border objects
        for new_core, neighbors in neighbors_of_new_core_neighbors.items():
            label = self._labels.get_label(new_core)
            for neighbor in neighbors:
                self._labels.set_label(neighbor, label)

    def _filter_core_objects_split_by_novelty(self, objects):
        new_cores = set()
        old_cores = set()

        for obj in objects:
            if obj.neighbor_count == self.min_pts:
                new_cores.add(obj)
            elif obj.neighbor_count > self.min_pts:
                old_cores.add(obj)

        return new_cores, old_cores

    def _get_neighbors_of_objects(self, objects, min_pts=0):
        neighbors = dict()

        for obj in objects:
            neighbors[obj] = \
                self._objects.get_neighbors(obj, self.eps, min_pts)

        return neighbors

    def _get_update_seeds(self, neighbors_of_new_core_neighbors):
        seeds = set()

        for neighbors in neighbors_of_new_core_neighbors.values():
            core_neighbors = [obj for obj in neighbors
                              if obj.neighbor_count >= self.min_pts]
            seeds.update(core_neighbors)

        return core_neighbors

    def _get_connected_components(self, objects):
        G = nx.Graph()

        for obj in objects:
            neighbors = \
                self._objects.get_neighbors(obj, self.eps, self.min_pts)
            for neighbor in neighbors:
                G.add_edge(obj, neighbor)

        return nx.connected_components(G)

    def _deletion(self, object_id):
        # Do lot of stuff, then:
        self._labels.delete_label(object_id)


class IncrementalDBSCANWarning(Warning):
    pass

# TODO notebook import fails now
