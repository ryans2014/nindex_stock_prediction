import unittest
from models import DataPreprocessor


class TestDataPreProcessor(unittest.TestCase):

    def test_single_stock_flow(self):
        ret = DataPreprocessor().load_from_raw_json("AAPL").expand().extract_sequence().split().get()
        self.assertEqual(len(ret), 6)
        self.assertEqual(len(ret[1]), 0)
        self.assertEqual(len(ret[3]), 0)


if __name__ == "__main__":
    unittest.main()
