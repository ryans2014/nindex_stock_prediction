from model import get_batch_input_array
import unittest


class TestPreprocessor(unittest.TestCase):

    def test_input_flow(self):
        ct = 0
        for x, y in get_batch_input_array(batch_size=50, sample_offset=50, input_length=19, dp_per_sma=4):
            self.assertEqual(x.shape, (50, 19, 5))
            self.assertEqual(y.shape, (50, 1))
            ct += 1
            if ct > 50:
                return
