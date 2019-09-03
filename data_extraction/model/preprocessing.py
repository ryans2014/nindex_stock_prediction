import os
import glob
import json
from configuration import work_dir
import utility
from data_extractor import convert_alphavantage_data_to_pandas
from model.util import normalize, get_sma, price_to_percentage
import numpy as np
import math


@utility.log_and_discard_exceptions
def _get_raw_json():
    json_path = os.path.join(work_dir, "sp500_data_raw_json\\*.json")
    for file in glob.glob(pathname=json_path, recursive=False):
        with open(file, 'r') as f:
            raw_obj = json.load(f)
            yield raw_obj


def _get_original_data_frame():
    for raw_js_obj in _get_raw_json():
        _, df = convert_alphavantage_data_to_pandas(raw_js_obj)
        yield df


def _get_expanded_data_frame():
    for df in _get_original_data_frame():
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


def _get_input_array(sample_offset, input_length=20, dp_per_sma=4):
    """
    :param sample_offset: number of days between training samples
    :param input_length: number of historical data points for every attributes
    :param dp_per_sma: controls the delta-t between data points, dp_per_sma=4 means 4 data points every SMA period
    :return: yield X: 2D np array with axis 1 the time sequence and axis 2 the attribute types
             yield Y: float
    """
    max_offset = math.ceil(200.0 / dp_per_sma)
    max_size = input_length * max_offset + 1
    for df in _get_expanded_data_frame():
        # target is the average price change in future 20 days
        idx = len(df) - 20

        while idx > max_size:
            # daily price
            start = idx - input_length - 1
            end = idx
            close = df["Close"][start:end].values
            close = price_to_percentage(close)
            average_percentage = np.abs(close).mean()
            close = close / average_percentage

            # sma5
            dp_interval = math.ceil(5.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma5 = df["SMA5"][start:end:2].values
            sma5 = price_to_percentage(sma5)
            average_percentage = np.abs(sma5).mean()
            sma5 = sma5 / average_percentage

            # sma20
            dp_interval = math.ceil(20.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma20 = df["SMA20"][start:end:5].values
            sma20 = price_to_percentage(sma20)
            average_percentage = np.abs(sma20).mean()
            sma20 = sma20 / average_percentage

            # sma100
            dp_interval = math.ceil(100.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma100 = df["SMA100"][start:end:25].values
            sma100 = price_to_percentage(sma100)
            average_percentage = np.abs(sma100).mean()
            sma100 = sma100 / average_percentage

            # sma200
            dp_interval = math.ceil(200.0 / dp_per_sma)
            start = idx - 1 - input_length * dp_interval
            end = idx
            sma200 = df["SMA200"][start:end:50].values
            sma200 = price_to_percentage(sma200)
            average_percentage = np.abs(sma200).mean()
            sma200 = sma200 / average_percentage

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
            y = (sma20_future / price_now - 1.0) * 100

            # next
            idx -= sample_offset
            yield x_array, y


def get_batch_input_array(batch_size, sample_offset, *args, **kwargs):
    """ Possible arguments:
        sample_offset: number of days between training samples
        input_length: number of historical data points for every attributes
        dp_per_sma: controls the delta-t between data points, dp_per_sma=4 means 4 data points every SMA period
    """
    x_list = []
    y_list = []
    for x1, y1 in _get_input_array(sample_offset, *args, **kwargs):
        x_list.append(x1.reshape((1, x1.shape[0], x1.shape[1])))
        y_list.append(y1)
        if len(y_list) == batch_size:
            x_array = np.concatenate(tuple(x_list))
            y_array = np.array(y_list).reshape(-1, 1)
            x_list.clear()
            y_list.clear()
            yield x_array, y_array


def plt_last_n(df, n):
    df[["Close", "SMA5", "SMA20", "SMA100", "SMA200"]][-n:].plot()

