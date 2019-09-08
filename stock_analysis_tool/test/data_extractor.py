from data_extractor import get_data
import unittest


class TestAlphavantageExtractor(unittest.TestCase):

    def test_get_data(self):
        ticker, df = get_data(ticker="AAPL", force_update=True, save=False)
        self.assertTrue(df is not None)
        self.assertTrue(ticker is not None)
        self.assertNotEqual(len(df), 0)

    def test_cache_write_and_read(self):
        ticker1, df1 = get_data(ticker="AAPL", force_update=True, save=True)
        ticker2, df2 = get_data(ticker="AAPL", force_update=False, save=False)
        self.assertEqual(ticker1, ticker2)
        self.assertEqual(len(df1), len(df2))


if __name__ == "__main__":
    unittest.main()
