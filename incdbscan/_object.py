from functools import cached_property


NodeId = int
ObjectId = int


class Object:
    def __init__(self, id_, min_pts):
        self.id: ObjectId = id_
        self.node_id: NodeId = None
        self.count = 1
        self.neighbors = {self}
        self.neighbor_count = 0
        self.min_pts = min_pts

    @cached_property
    def is_core(self):
        # Note that this property is only valid during deletion
        return self.neighbor_count >= self.min_pts

    def _clear_is_core_cache(self):
        self.__dict__.pop('is_core', None)

    def __repr__(self):
        return f'{self.id}_'
