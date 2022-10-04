import unittest
import time_machine as tm
import numpy as np
import torch


class TestDataset(unittest.TestCase):
    def test_with_data(self):
        raw = tm.read_csv('./test/test_data.csv')
        history = tm.TradingHistory(raw, window_past=16, window_future=5)

        t_now, t_pas, t_fut = history[0]

        assert t_pas[-1][3] == t_now[-1][3]
        assert torch.all(t_pas[-1] != t_fut[0])
        assert len(t_pas) == 16
        assert len(t_fut) == 5