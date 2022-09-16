import unittest
import time_machine as tm


class TestConsistency(unittest.TestCase):
    def test_intraday(self):
        raw = tm.read_csv('./example_data/SPG_2020_2020.txt')
        history = tm.History(raw)
        
        for i in range(len(history)):
            intra = history.get_intraday_idx(i)
            o, h, l, c, v = history.get_day_idx(i)
            H = max(intra[:, 1])
            L = min(intra[:, 2])

            self.assertEqual(H, h)
            self.assertEqual(L, l)