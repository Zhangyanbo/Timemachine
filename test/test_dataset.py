import unittest
import time_machine as tm
import numpy as np


class TestDataset(unittest.TestCase):
    def test_iter_range(self):
        raw = tm.read_csv('./example_data/SPG_2020_2020.txt')
        history = tm.TradingHistory(raw)

        for i in range(len(history)):
            tn, tp, tf = history[i]