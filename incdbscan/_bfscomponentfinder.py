from collections import defaultdict
from typing import (
    Dict,
    List
)

import rustworkx as rx
from rustworkx.visit import (
    BFSVisitor,
    PruneSearch
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
    # Note that it could even faster if the traversal terminted when all of
    # the next nodes to be visited are linked to the same seed object -- this
    # means that all but one component are traversed completely and they can
    # be split away.

    def __init__(self, graph):
        self.graph: rx.PyGraph = graph  # graph of Objects  # pylint: disable=no-member
        self.seed_to_component: Dict[NodeId, List[Object]] = defaultdict(set)
        self._node_to_seed: Dict[NodeId, NodeId] = defaultdict(int)

    def discover_vertex(self, vertex_node_id):
        # If this is the first time discovering a node then the node itself
        # will be its own seed. This is the way we keep track of singleton
        # nodes (i.e., ones without edges).

        if vertex_node_id not in self._node_to_seed:
            self._node_to_seed[vertex_node_id] = vertex_node_id
            self.seed_to_component[vertex_node_id].add(self.graph[vertex_node_id])

        # If the node does not represent a core object then we don't want
        # traversal to go in that direction.

        if not self.graph[vertex_node_id].is_core:
            raise PruneSearch

    def tree_edge(self, edge):
        source_node_id, target_node_id, _ = edge

        # The target of the edge is a new node we see for the first time. Its
        # seed will be the seed of the source.

        seed = self._node_to_seed[source_node_id]
        self._node_to_seed[target_node_id] = seed
        self.seed_to_component[seed].add(self.graph[target_node_id])

    def non_tree_edge(self, edge):
        source_node_id, target_node_id, _ = edge

        # A non-tree edge is the case of merge, that is, when two components
        # with different seeds meet. However, we only merge them if the target
        # represents a core object in the graph (i.e., dense connection).

        source_seed = self._node_to_seed[source_node_id]
        target_seed = self._node_to_seed[target_node_id]
        different_seeds = source_seed != target_seed

        if different_seeds and self.graph[target_node_id].is_core:
            if source_seed > target_seed:
                self._node_to_seed[target_node_id] = source_seed
            else:
                self._node_to_seed[source_node_id] = target_seed

        seed = self._node_to_seed[target_node_id]
        self.seed_to_component[seed].add(self.graph[target_node_id])
