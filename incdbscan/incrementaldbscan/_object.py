ObjectId = str


class _Object:
    def __init__(self, value, id_):
        self.id: ObjectId = id_
        self.value = value
        self.count = 1
        self.neighbors = set([self])

    @property
    def neighbor_count(self):
        return sum([neighbor.count for neighbor in self.neighbors])

    def __repr__(self):
        return f'{self.value}_'
