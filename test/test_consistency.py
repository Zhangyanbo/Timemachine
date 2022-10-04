import unittest
import time_machine as tm


class TestConsistency(unittest.TestCase):
    def test_intraday(self):
        raw = tm.read_csv('./example_data/SPG_2020_2020.txt')
        history = tm.History(raw)
        
        for i in range(len(history)):
            intra = history.get_full_intraday_idx(i)
            o, h, l, c, v = history.get_day_idx(i)
            H = intra[:, 1].max()
            L = intra[:, 2].min()

            if H != h or L != l:
                print(f'i = {i}, H = {H}, h = {h}, L = {L}, l = {l}')
            
            assert H == h
            assert L == l