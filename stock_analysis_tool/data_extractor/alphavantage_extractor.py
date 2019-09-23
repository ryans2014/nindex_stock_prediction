import requests
import utility
from configuration import get_config
import logging
from .data_extractor import DataExtractor


class AlphavantageExtractor(DataExtractor):

    _cool_down_time = 60. / get_config("alphavantage", "limit_per_min") + 1.

    def __init__(self):
        self._token = get_config("alphavantage", "api_key")
        logging.info("Creating Alphavantage extractor, _cool_down_time is %d" % AlphavantageExtractor._cool_down_time)
        logging.info("Creating Alphavantage extractor, _token_id is %s" % self._token)

    def _get_eod_query_url(self, ticker: str, get_full_data: bool) -> str:
        """
        :param ticker: stock ticker string
        :param get_full_data: False is only want the last 100 day result, True if want the full historical data
        :return: a url string to get the required data
        """
        url_format = "https://www.alphavantage.co/query?function=%s&outputsize=%s&symbol=%s&apikey=%s"
        fun = "TIME_SERIES_DAILY_ADJUSTED"
        output_size = ['compact', 'full'][get_full_data]
        return url_format % (fun, output_size, ticker, self._token)

    @staticmethod
    @utility.frequency_limiter(_cool_down_time)
    def _get_eod_data(url: str):
        """
        :param url: url to get data
        :param get_full_data: False is only want the last 100 day result, True if want the full historical data
        :return: a dictionary that contains the requested data, return None if failure
        """
        try:
            response = requests.get(url)
        except Exception as excp:
            logging.warning("Exception caught during get method, URL = %s, Exception = %s" % (url, str(excp)))
            raise ValueError("SYMBOL_FETCH_ERROR")
        if response.status_code is not 200:
            logging.warning("Http response status code = %d, URL = %s" % (response.status_code, url))
            raise ValueError("SYMBOL_FETCH_ERROR")
        data = response.json()
        if "Error Message" in data:
            logging.warning("Http response got error message (%s)" % str(data["Error Message"]))
            raise ValueError("SYMBOL_FETCH_ERROR")
        if "Time Series (Daily)" not in data or len(data["Time Series (Daily)"]) == 0:
            logging.warning("Http response has zero entries")
            raise ValueError("SYMBOL_FETCH_ERROR")

        logging.info("Data extraction from remote successful. (%s)" % url)
        return data

    def extract(self, ticker: str, get_full_data: bool = True) -> dict:
        """
        returns a json-type dict object
        """
        url = self._get_eod_query_url(ticker, get_full_data)
        return AlphavantageExtractor._get_eod_data(url)
