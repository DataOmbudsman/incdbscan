NodeId = int
ObjectId = bytes

import numpy as np

from ._utils import decode_


class Object:
    def __init__(self, id_):
        self.id: ObjectId = id_
        self.node_id: NodeId = None
        self.count = 1
        self.neighbors = {self}
        self.neighbor_count = 0

    def __repr__(self):
        return f"{self._decode_id()}"

    def _decode_id(self):
        return decode_(self.id)

    def get_value(self):
        return np.stack([self._decode_id()] * self.count)
