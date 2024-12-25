NodeId = int
ObjectId = int


class Object:
    def __init__(self, id_):
        self.id: ObjectId = id_
        self.node_id: NodeId = None
        self.count = 1
        self.neighbors = {self}
        self.neighbor_count = 0

    def __repr__(self):
        return f'{self.id}_'
