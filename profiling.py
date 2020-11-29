from datetime import datetime
from pathlib import Path

import numpy as np
from line_profiler import LineProfiler

from incdbscan import IncrementalDBSCAN
from incdbscan._inserter import Inserter
from incdbscan._deleter import Deleter


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'incdbscan' / 'tests' / 'data'


def test1():
    data_set = '2d-20c-no0.dat'
    data = np.loadtxt(DATA_PATH / data_set)[:, 0:2]

    algo = IncrementalDBSCAN(eps=1)
    algo.insert(data)
    algo.delete(data)


def test2():
    data_set = 't4.8k.dat'
    data = np.loadtxt(DATA_PATH / data_set)[:2000]

    algo = IncrementalDBSCAN(eps=10)
    algo.insert(data)
    algo.delete(data)


def print_profile(test):
    profiler = LineProfiler()
    profiler.add_module(Inserter)
    profiler.add_module(Deleter)

    wrapper = profiler(test)
    wrapper()

    timestamp = str(datetime.now())[:19]
    filename = f'{timestamp}_{test.__name__}.txt'
    profile_path = BASE_PATH / 'profiling' / filename

    with open(profile_path, 'w') as f:
        profiler.print_stats(stream=f)


if __name__ == "__main__":
    for test in [test1, test2]:
        print(f'{datetime.now()} Creating profile for {test.__name__} ...')
        print_profile(test)
