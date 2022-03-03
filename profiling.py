import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from line_profiler import LineProfiler

from incdbscan import IncrementalDBSCAN
from incdbscan._deleter import Deleter
from incdbscan._inserter import Inserter
from incdbscan.tests.testutils import read_text_data_file_from_url


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'incdbscan' / 'tests' / 'data'


def test1():
    # This is equivalent to the 2d-20c-no0 data set by Handl, J.
    # Also available from:
    # https://personalpages.manchester.ac.uk/staff/Julia.Handl/generators.html

    url = (
        'https://gitlab.christianhomberg.me/chr_96er/MOCK-PESA-II/raw/'
        '54572f1f371a3e8f59999c40957df1485acad8b5/MOCK/data/MOCKDATA/'
        '2d-20c-no0.dat'
    )
    data = read_text_data_file_from_url(url)[:, 0:2]

    algo = IncrementalDBSCAN(eps=1)
    algo.insert(data)
    algo.delete(data)


def test2():
    # This is equivalent to the t4.8k data set from the Chameleon collection
    # by Karypis, G. et al. Also available from:
    # http://glaros.dtc.umn.edu/gkhome/cluto/cluto/download

    url = (
        'https://raw.githubusercontent.com/yeahia2508/ml-examples/'
        'master/Data/clustering/chameleon/t4.8k.txt'
    )
    data = read_text_data_file_from_url(url)[:2000]

    algo = IncrementalDBSCAN(eps=10)
    algo.insert(data)
    algo.delete(data)


def print_profile(test, tag=''):
    profiler = LineProfiler()
    profiler.add_module(Inserter)
    profiler.add_module(Deleter)

    wrapper = profiler(test)
    wrapper()

    timestamp = str(datetime.now())[:19]
    filename = f'{timestamp}_{test.__name__}{tag}.txt'
    profile_path = BASE_PATH / 'profiling' / filename

    with open(profile_path, 'w') as f:
        profiler.print_stats(stream=f)


if __name__ == "__main__":
    tag = '_' + sys.argv[1] if len(sys.argv) > 1 else ''
    for test in [test1, test2]:
        print(f'{datetime.now()} Creating profile for {test.__name__} ...')
        print_profile(test, tag)
