import requests
from data_extractor.frequency_limiter import frequency_limiter
from data_extractor.configuration import get_config
import logging


class AlphavantageExtractor:

    _cool_down_time = 60. / get_config("alphavantage", "limit_per_min") + 1.
    _token = get_config("alphavantage", "api_key")

    def __init__(self):
        logging.info("Creating Alphavantage extractor, _cool_down_time is %d" % AlphavantageExtractor._cool_down_time)
        logging.info("Creating Alphavantage extractor, _token_id is %d" % AlphavantageExtractor._token)

    @staticmethod
    def _get_eod_query_url(ticker: str, get_full_data: bool) -> str:
        """
        :param ticker: stock ticker string
        :param get_full_data: False is only want the last 100 day result, True if want the full historical data
        :return: a url string to get the required data
        """
        url_format = "https://www.alphavantage.co/query?function=%s&outputsize=%s&symbol=%s&apikey=%s"
        fun = "TIME_SERIES_DAILY_ADJUSTED"
        output_size = ['compact', 'full'][get_full_data]
        return url_format % (fun, output_size, ticker, AlphavantageExtractor._token)

    @staticmethod
    @frequency_limiter(_cool_down_time)
    def _get_eod_data(ticker: str, get_full_data=False):
        """
        :param ticker: stock ticker string
        :param get_full_data: False is only want the last 100 day result, True if want the full historical data
        :return: a dictionary that contains the requested data, return None if failure
        """
        url = AlphavantageExtractor._get_eod_query_url(ticker, get_full_data)
        try:
            response = requests.get(url)
        except Exception as excp:
            logging.WARNING("Exception caught during get method, URL = %s, Exception = %s" % (url, str(excp)))
            return None
        if response.status_code is not 200:
            return {}
        data = response.json()
        if "Error Message" in data:
            return {}
        if "Time Series (Daily)" not in data or len(data["Time Series (Daily)"]):
            return {}
        return data

    def _process_data(self, data: dict):
        """
        :param data:
        :return:
        """
        print("Successfully obtained ticker %s for %d days"
              % (data["Meta Data"]["Symbol"], len(data["Time Series (Daily)"])))

    def batch_extract(self, ticker_list: list, get_full_data: bool) -> None:
        for ticker in ticker_list:
            data = self._get_eod_data(ticker, get_full_data)
            self._process_data(data)




