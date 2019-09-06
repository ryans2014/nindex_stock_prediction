from utility import frequency_limiter
from time import time
import unittest

waiting_time = 0.5


@frequency_limiter(waiting_time)
def get_time() -> float:
    return time()


@frequency_limiter(0.0)
def forward_argument(arg1, arg2, arg3):
    return arg1, arg2, arg3


class TestFrequencyLimiter(unittest.TestCase):

    def test_frequency(self):
        get_time()
        t2 = get_time()
        t3 = get_time()
        delta_t = t3 - t2
        self.assertFalse(delta_t > waiting_time + 0.1 or delta_t < waiting_time - 0.1)

    def test_name(self):
        self.assertEqual(get_time.__name__, "get_time")

    def test_argument(self):
        ret = forward_argument(1, arg3=3, arg2=2)
        self.assertEqual(ret, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
