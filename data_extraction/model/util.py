from sklearn.preprocessing import StandardScaler
import numpy as np


def normalize(np_arr):
    """
    :param np_arr: 1D np array
    :return: 1D np array, normalized
    """
    return StandardScaler(copy=False).fit_transform(np_arr.reshape(-1, 1)).reshape(-1)


def get_sma(np_arr, period_list):
    """
    :param np_arr: 1d np array
    :param period_list: list of int, moving average period
    :return: list of ndarray with same size, pad NAN in front
    """
    cum = np.cumsum(np_arr)
    ret = []
    for period in period_list:
        sma_array = (cum[period:] - cum[:-period]) / float(period)
        nan_array = np.full(period, np.nan)
        full_array = np.concatenate((nan_array, sma_array))
        ret.append(full_array)
    return ret


def price_to_percentage(np_arr):
    return (np_arr[1:] / np_arr[:-1] - 1) * 100
