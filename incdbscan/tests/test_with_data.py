import pickle

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from incdbscan import IncrementalDBSCAN
from testutils import (
    are_lists_isomorphic,
    read_chameleon_data,
    read_handl_data
)


@pytest.mark.slow
def test_same_results_as_sklearn_dbscan():
    EPS = 1
    MIN_PTS = 5

    data = read_handl_data()
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_PTS)
    labels_dbscan = dbscan.fit_predict(data)

    incdbscan = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels_incdbscan_1 = incdbscan.insert(data).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_1)

    labels_incdbscan_2 = \
        incdbscan.insert(data).delete(data).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_2)

    np.random.seed(123)
    noise = np.random.uniform(-14, 14, (1000, 2))
    labels_incdbscan_3 = \
        incdbscan.insert(noise).delete(noise).get_cluster_labels(data)
    assert are_lists_isomorphic(labels_dbscan, labels_incdbscan_3)


@pytest.mark.slow
def test_pickling_works_and_preserves_model_functionality():
    EPS = 15
    MIN_PTS = 40

    data = read_chameleon_data()  # total size: 8000 rows
    first_batch_end = 7000

    incdbscan1 = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    labels1 = incdbscan1.insert(data).get_cluster_labels(data)

    incdbscan2 = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    incdbscan2.insert(data[:first_batch_end])

    serialized = pickle.dumps(incdbscan2)
    incdbscan2_reconstructed = pickle.loads(serialized)

    incdbscan2_reconstructed.insert(data[first_batch_end:])
    labels2 = incdbscan2_reconstructed.get_cluster_labels(data)

    assert are_lists_isomorphic(labels1, labels2)
