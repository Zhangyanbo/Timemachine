import unittest
import time_machine as tm
import numpy as np


class TestConsistency(unittest.TestCase):
    def test_past(self):
        raw = tm.read_csv('./example_data/SPG_2020_2020.txt')
        history = tm.History(raw)
        
        path = history.get_day_past_idx(idx=5, dt=3)

        path_target = history.df_day.iloc[[3, 4, 5]][['O', 'H', 'L', 'C', 'V']].values.astype(np.float64)

        assert np.all(path == path_target)
    
    def test_future(self):
        raw = tm.read_csv('./example_data/SPG_2020_2020.txt')
        history = tm.History(raw)
        
        path = history.get_day_future_idx(idx=1, dt=3)

        path_target = history.df_day.iloc[[1, 2, 3]][['O', 'H', 'L', 'C', 'V']].values.astype(np.float64)

        assert np.all(path == path_target)