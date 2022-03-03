import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from incdbscan import IncrementalDBSCAN
from testutils import (
    are_lists_isomorphic,
    read_text_data_file_from_url
)


@pytest.mark.slow
def test_same_results_as_sklearn_dbscan():
    # This is equivalent to the 2d-20c-no0 data set by Handl, J.
    # Also available from:
    # https://personalpages.manchester.ac.uk/staff/Julia.Handl/generators.html

    url = (
        'https://gitlab.christianhomberg.me/chr_96er/MOCK-PESA-II/raw/'
        '54572f1f371a3e8f59999c40957df1485acad8b5/MOCK/data/MOCKDATA/'
        '2d-20c-no0.dat'
    )
    data = read_text_data_file_from_url(url)[:, 0:2]

    EPS = 1
    MIN_PTS = 5

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
