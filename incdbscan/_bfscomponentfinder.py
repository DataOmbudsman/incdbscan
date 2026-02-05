from collections import (
    defaultdict,
    deque
)
from typing import (
    Dict,
    Set
)

import rustworkx as rx
from rustworkx.visit import (
    BFSVisitor,
    PruneSearch,
    StopSearch
)

from ._object import (
    NodeId,
    Object
)


class BFSComponentFinder(BFSVisitor):

    # Traverse the objects in a BFS manner to find those components of
    # objects that need to be split away. A component here is a group of
    # objects that all can be linked to the same seed object. Starting from
    # the seed objects, expand the graph by adding neighboring objects.
    # The traversal termintes when all of the next nodes to be visited are
    # linked to the same seed object -- this means that all but one component
    # are traversed completely and they can be split away.

    def __init__(self, graph):
        self._graph: rx.PyGraph = graph  # graph of Objects  # pylint: disable=no-member
        self._seed_to_component: Dict[NodeId, Set[Object]] = defaultdict(set)
        self._node_to_seed: Dict[NodeId, NodeId] = {}
        self._queue = deque()

    def _preprocess(self, seeds):
        # We create a fake node in the graph that is connected to all seeds to
        # to implement multi-seed BFS.

        origin_object = Object('ORIGIN', 0)
        self._origin_node_id = self._graph.add_node(origin_object)
        edges_from_origin = [(self._origin_node_id, seed_node_id, None)
                             for seed_node_id in seeds]
        self._graph.add_edges_from(edges_from_origin)

    def _same_seeds(self):
        iterator = iter(self._queue)
        first_obj = next(iterator)
        first_seed = self._node_to_seed[first_obj]

        for obj in iterator:
            seed = self._node_to_seed[obj]
            if seed != first_seed:
                return False
        return True

    def discover_vertex(self, vertex_node_id):
        # If this is the first time discovering a node then the node itself
        # will be its own seed. This is the way we keep track of singleton
        # nodes (i.e., ones without edges).

        if vertex_node_id not in self._node_to_seed:
            self._node_to_seed[vertex_node_id] = vertex_node_id
            self._seed_to_component[vertex_node_id].add(self._graph[vertex_node_id])

        # If the node does not represent a core object then we don't want
        # traversal to go in that direction.

        if self._graph[vertex_node_id].is_core:
            self._queue.append(vertex_node_id)
        else:
            raise PruneSearch

    def finish_vertex(self, _):
        _ = self._queue.popleft()
        if self._same_seeds():
            raise StopSearch

    def tree_edge(self, edge):
        source_node_id, target_node_id, _ = edge

        # The target of the edge is a new node we see for the first time. Its
        # seed will be the seed of the source. The source being the origin node
        # is an exception.

        source_is_origin = source_node_id == self._origin_node_id
        target_seed = (target_node_id if source_is_origin
                       else self._node_to_seed[source_node_id])

        self._node_to_seed[target_node_id] = target_seed
        self._seed_to_component[target_seed].add(self._graph[target_node_id])

    def gray_target_edge(self, edge):
        source_node_id, target_node_id, _ = edge

        # A gray target edge is the case of merge, that is, when two components
        # with different seeds meet. However, we only merge them if the target
        # represents a core object in the graph (i.e., dense connection).

        source_seed = self._node_to_seed[source_node_id]
        target_seed = self._node_to_seed[target_node_id]
        different_seeds = source_seed != target_seed

        if different_seeds and self._graph[target_node_id].is_core:
            # Let the seed of the source be the unified seed for both
            # components. The seed of the target is discarded.
            objects_to_merge = self._seed_to_component[target_seed]
            for obj in objects_to_merge:
                self._node_to_seed[obj.node_id] = source_seed
            self._seed_to_component[source_seed].update(objects_to_merge)
            del self._seed_to_component[target_seed]

    def _postprocess(self):
        # Delete data of fake node
        self._graph.remove_node(self._origin_node_id)
        del self._seed_to_component[self._origin_node_id]

        # Discard the component not traversed
        remaining_node = self._queue.popleft()
        remaining_seed = self._node_to_seed[remaining_node]
        del self._seed_to_component[remaining_seed]

    def find_components(self, seeds):
        self._preprocess(seeds)
        rx.bfs_search(self._graph, [self._origin_node_id], self)
        self._postprocess()
        return self._seed_to_component.values()
