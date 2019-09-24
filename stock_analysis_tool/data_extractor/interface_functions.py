import utility
import logging
import pandas as pd
import json
import asyncio
from .alphavantage_extractor import AlphavantageExtractor
from .cache_extractor import CacheExtractor, CacheWriter


def get_data(ticker: str, force_update=False, save=True):
    """
    :param ticker: str, ticker name, like "AAPL"
    :param force_update: bool, if True, force to extract from web source
    :param save: if save to cache file
    :return: (str, DataFrame) tuple
    """
    ret_obj = {}
    if not force_update:
        ret_obj = _cache_extractor.extract(ticker)
    if ret_obj == {}:
        ret_obj = _web_extractor.extract(ticker)
    if save:
        _cache_writer.write(ticker, ret_obj)
    return _convert_alphavantage_data_to_pandas(ret_obj)


async def get_data_async(ticker: str):
    """
    :param ticker: str, ticker name, like "AAPL"
    :return: (str, DataFrame) tuple
    """
    ret_obj = await _web_extractor.extract_async(ticker)
    return _convert_alphavantage_data_to_pandas(ret_obj)


def _convert_alphavantage_data_to_pandas(js_obj):
    """
    :param js_obj: json file path string or json dict object
    :return: (ticker string, pandas DataFrame)
    """
    if type(js_obj) is str:
        with open(js_obj) as fp:
            js_obj = json.load(fp)

    ticker = js_obj['Meta Data']['2. Symbol']
    js_obj = js_obj['Time Series (Daily)']
    logging.info("process_data_to_get_closing_price: %s" % ticker)
    logging.getLogger().handlers[0].flush()
    ls_obj = []
    for date, price in js_obj.items():
        int_date = utility.date.date_to_int(date)
        adjusted_close = float(price['5. adjusted close'])
        volume = float(price['6. volume'])
        ls_obj.append([int_date, adjusted_close, volume])
    df = pd.DataFrame(data=ls_obj, columns=["Date", "Close", "Volume"])
    df.sort_values(by="Date", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return ticker, df


_web_extractor = AlphavantageExtractor()
_cache_extractor = CacheExtractor()
_cache_writer = CacheWriter()
