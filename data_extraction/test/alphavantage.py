from data_extractor.alphavantage_extractor import AlphavantageExtractor
import unittest


class TestAlphavantageExtractor(unittest.TestCase):

    def setUp(self) -> None:
        self.extractor = AlphavantageExtractor()

    def test_valid_ticker(self):
        obj = self.extractor.batch_extract(['AAPL'], True)
        self.assertEqual(type(obj), list)
        self.assertEqual(len(obj), 1)
        obj = obj[0]
        self.assertFalse("Error Message" in obj)
        self.assertTrue("Time Series (Daily)" in obj)
        self.assertEqual(type(obj), dict)
        self.assertGreater(len(obj["Time Series (Daily)"]), 0)

    def test_invalid_ticker(self):
        obj = self.extractor.extract('AAPLABC', True)
        self.assertEqual(len(obj), 0)
        self.assertEqual(type(obj), dict)

    def tearDown(self) -> None:
        self.extractor = None


if __name__ == "__main__":
    unittest.main()
