import os
import glob
import json
from configuration import work_dir
import utility
from data_extractor import convert_alphavantage_data_to_pandas
from model.math_util import normalize, get_sma, price_to_percentage
import numpy as np
import math
import functools
import logging


@utility.log_and_discard_exceptions
def _get_raw_json(endless=False):
    json_path = os.path.join(work_dir, "sp500_data_raw_json\\*.json")
    while True:
        for file in glob.glob(pathname=json_path, recursive=False):
            with open(file, 'r') as f:
                raw_obj = json.load(f)
                yield raw_obj
        if not endless:
            break


def _get_original_data_frame(**kwargs):
    for raw_js_obj in _get_raw_json(**kwargs):
        _, df = convert_alphavantage_data_to_pandas(raw_js_obj)
        yield df


def _get_expanded_data_frame(**kwargs):
    for df in _get_original_data_frame(**kwargs):
        if len(df) < 500:
            continue

        # log + normalize volume data
        # df["Volume"] = df["Volume"].map(np.log)
        # df["Volume"] = normalize(df["Volume"].values)

        # add sma
        sma_length = [5, 20, 100, 200]
        price = df["Close"].values
        sma_price = get_sma(price, sma_length)
        df["SMA5"] = sma_price[0]
        df["SMA20"] = sma_price[1]
        df["SMA100"] = sma_price[2]
        df["SMA200"] = sma_price[3]

        df.dropna(inplace=True)
        yield df


def _get_input_array(sample_offset, input_length,
                     date_cutoff=-1, month_cutoff=-1, year_cutoff=-1,
                     **kwargs):
    dp_per_sma = 4

    date_cutoff = max(date_cutoff, month_cutoff * 21, year_cutoff * 253)
    if date_cutoff < 1000:
        date_cutoff = -1

    max_offset = math.ceil(200.0 / dp_per_sma)
    max_size = input_length * max_offset + 1
    for df in _get_expanded_data_frame(**kwargs):
        # target is the average price change in future 20 days
        idx = len(df) - 20

        while idx > max_size:
            # daily price
            start = idx - input_length - 1
            end = idx
            close = df["Close"][start:end].values
            close = price_to_percentage(close) * 0.25
            close = np.tanh(close)

            # sma5
            dp_interval = math.ceil(5.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma5 = df["SMA5"][start:end:2].values
            sma5 = price_to_percentage(sma5) * 0.5 / 2.0
            sma5 = np.tanh(sma5)

            # sma20
            dp_interval = math.ceil(20.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma20 = df["SMA20"][start:end:5].values
            sma20 = price_to_percentage(sma20) / 5.0
            sma20 = np.tanh(sma20)

            # sma100
            dp_interval = math.ceil(100.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma100 = df["SMA100"][start:end:25].values
            sma100 = price_to_percentage(sma100) * 2.0 / 25.0
            sma100 = np.tanh(sma100)

            # sma200
            dp_interval = math.ceil(200.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma200 = df["SMA200"][start:end:50].values
            sma200 = price_to_percentage(sma200) * 2.5 / 50.0
            sma200 = np.tanh(sma200)

            # combine matrix
            close.reshape(-1, 1)
            x_array = np.concatenate((close.reshape(-1, 1),
                                      sma5.reshape(-1, 1),
                                      sma20.reshape(-1, 1),
                                      sma100.reshape(-1, 1),
                                      sma200.reshape(-1, 1)), axis=1)

            # get target
            sma20_future = df["SMA20"][idx + 19]
            price_now = df["Close"][idx - 1]
            y = (sma20_future / price_now - 1.0) * 100.0

            # mapping to around (-2, 2) region
            y = np.tanh(y / 10.0)

            # check shape and nan
            if x_array.shape != (input_length, 5):
                logging.error("Incorrect x_array shape:" + str(x_array.shape))
            elif np.count_nonzero(np.isnan(x_array)) > 0:
                logging.error("Found NAN in x_array")
            elif y == np.nan:
                logging.error("y = NAN")
            else:
                yield x_array, y

            # next
            idx -= sample_offset
            if date_cutoff > 0 and idx < len(df) - date_cutoff:
                break


def get_batch_input_array(batch_size, sample_offset, input_length=20, **kwargs):
    """
    :param batch_size: number of samples in a batch, if negative, generate all samples
    :param sample_offset: number of days between training samples
    :param input_length: int, number of historical data points for every attributes
    :param kwargs: supported key-value types in kwargs
                dp_per_sma: int, controls delta-t between data points, dp_per_sma=4 means 4 data points every SMA period
                date_cutoff: int, only consider the recent _ days of data
                month_cutoff, year_cutoff: int, similiar
                endless: bool
    :return: yield a tuple (x_array, y_array) as batch data
        x_array.shape = (batch_size, 20, 5)
        y_array.shape = (batch_size, 1)
    """
    x_list = []
    y_list = []
    for x1, y1 in _get_input_array(sample_offset, input_length, **kwargs):
        x_list.append(x1.reshape((1, x1.shape[0], x1.shape[1])))
        y_list.append(y1)
        if len(y_list) == batch_size:
            x_array = np.concatenate(tuple(x_list))
            y_array = np.array(y_list).reshape(-1, 1)
            x_list.clear()
            y_list.clear()
            yield x_array, y_array
    if batch_size < 0:
        x_array = np.concatenate(tuple(x_list))
        y_array = np.array(y_list).reshape(-1, 1)
        yield x_array, y_array


def batch_generator_creator(batch_size, sample_offset, **kwargs):
    """
    :return: generator function get_batch_input_array with given inputs
    """
    fun = functools.partial(get_batch_input_array, batch_size, sample_offset, **kwargs)
    fun = functools.wraps(get_batch_input_array)(fun)
    return fun

