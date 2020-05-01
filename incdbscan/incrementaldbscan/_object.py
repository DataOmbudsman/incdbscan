ObjectId = str
ClusterLabel = int

CLUSTER_LABEL_UNCLASSIFIED: ClusterLabel = -2
CLUSTER_LABEL_NOISE: ClusterLabel = -1
CLUSTER_LABEL_FIRST_CLUSTER: ClusterLabel = 0


class _Object:
    def __init__(self, value, id_):
        self.id: ObjectId = id_
        self.value = value

        self.count = 1
        self.label: ClusterLabel = CLUSTER_LABEL_UNCLASSIFIED
        self.neighbors = set([self])

    @property
    def neighbor_count(self):
        return sum([neighbor.count for neighbor in self.neighbors])
