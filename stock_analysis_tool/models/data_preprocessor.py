import os
import glob
import math
import logging
import csv
import numpy as np
import utility
import pickle as pk
from data_extractor import get_data
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
        How to Use:
        x_train, y_train, x_test, y_test = DataPreprocessor().load_from_xx(options)
                                                             .expand(options)
                                                             .extract_sequence(options)
                                                             .split(options)
                                                             .get(options)
        Or shortcut methods:
        x_train, y_train, x_test, y_test = DataProcessor()(ticker="name") to get single ticker for plotting

    """
    _cache_folder_name = "raw_json_cache"

    def __init__(self):
        self._dataframe = []
        self._skip_to_end = False
        self._single_ticker_call = False
        self._x_train = []
        self._x_test = []
        self._y_train = []
        self._y_test = []
        self._date = []
        self._price_now = []

    def load_from_raw_json(self, single_ticker=None, force_update=False, save=True):
        """
        :param single_ticker: str, ticker of the stock you need
        :param force_update: bool, force to fetch from remote
        :param save: bool whether to save to json file
        :return:
        """
        if type(single_ticker) is str:
            self._single_ticker_call = True
            if single_ticker.endswith("json"):
                single_ticker = single_ticker[:-5]
            _, df = get_data(ticker=single_ticker, force_update=force_update, save=save)
            self._dataframe = [df]
            return self
        json_path = os.path.join(os.path.join(os.getcwd(), DataPreprocessor._cache_folder_name), "*.json")
        self._dataframe = []
        for path in glob.glob(pathname=json_path, recursive=False):
            ticker = os.path.split(path)[-1][:-5].lower()
            _, df = get_data(ticker=ticker, force_update=force_update, save=save)
            if df is None:
                continue
            self._dataframe.append(df)
        return self

    def load_from_csv(self, csv_path: str):
        """
        :param csv_path: csv is a column of ticker strings
        :return:
        """
        self._dataframe = []
        with open(csv_path, "r") as fp:
            cr = csv.reader(fp)
            for line in cr:
                ticker = line[0]
                _, df = get_data(ticker=ticker, force_update=False, save=True)
                if df is None:
                    continue
                self._dataframe.append(df)
        return self

    def load_from_pickle(self):
        with open("x_train.pk", "rb") as fp:
            self._x_train = pk.load(fp)
        with open("x_test.pk", "rb") as fp:
            self._x_test = pk.load(fp)
        with open("y_train.pk", "rb") as fp:
            self._y_train = pk.load(fp)
        with open("y_test.pk", "rb") as fp:
            self._y_test = pk.load(fp)
        self._skip_to_end = True
        return self

    def expand(self):
        """
        :return: self
        Side Effect: add SMA features to Dataframe
        """
        if self._skip_to_end:
            return self

        new_data = []
        for df in self._dataframe:
            if len(df) < 500:
                continue
            # add sma
            price = df["Close"].values
            sma_price = utility.math_functions.get_sma(price)
            df["SMA5"] = sma_price[0]
            df["SMA20"] = sma_price[1]
            df["SMA100"] = sma_price[2]
            df["SMA200"] = sma_price[3]
            df.dropna(inplace=True)
            new_data.append(df)

        self._dataframe = new_data
        return self

    def extract_sequence(self,
                         sample_offset=5,
                         input_length=20,
                         year_cutoff=-1):
        """
        :param: sample_offset, int, number of days between neighbor samples
        :param: input_length, int, number of data point in time sequence, time period between data points may vary
            Final output size should be X: (n-sample, input_length, num_features)
        :param: year_cutoff, int, only recent n years are included
        Data normalization/regularization/clean-up are performed here.
        Price to Percentage transformation is done here.
        """
        if self._skip_to_end:
            return self

        # parameter that controls the number of days between two point in time sequence
        # 4 means for 200SMA, time serie points are 50 days apart to each other. For 20SMA, 5 days apart.
        dp_per_sma = 4

        for i, df in enumerate(self._dataframe):
            logging.info("extract_sequence %d/%d" % (i, len(df)))

            # starting bound
            starting_bound = input_length * math.ceil(200.0 / dp_per_sma) + 200 + 1
            if year_cutoff > 0:
                starting_bound = max(starting_bound, len(df) - year_cutoff * 253)

            # ending bound
            ending_bound = len(df) - 20
            if self._single_ticker_call:
                ending_bound = len(df)

            # adjust to make last day the last sample
            starting_bound += (ending_bound - starting_bound - 1) % sample_offset

            # starting iteration
            for idx in range(starting_bound, ending_bound, sample_offset):

                # daily price
                start = idx - input_length - 1
                end = idx
                close = df["Close"].iloc[start:end].values
                close = utility.math_functions.price_to_percentage(close) * 0.25
                close = np.tanh(close)

                # sma5
                dp_interval = math.ceil(5.0 / dp_per_sma)
                start = idx - 1 - input_length * dp_interval
                end = idx
                sma5 = df["SMA5"].iloc[start:end:dp_interval].values
                sma5 = utility.math_functions.price_to_percentage(sma5) * 0.5 / dp_interval
                sma5 = np.tanh(sma5)

                # sma20
                dp_interval = math.ceil(20.0 / dp_per_sma)
                start = idx - 1 - input_length * dp_interval
                end = idx
                sma20 = df["SMA20"].iloc[start:end:dp_interval].values
                sma20 = utility.math_functions.price_to_percentage(sma20) / dp_interval
                sma20 = np.tanh(sma20)

                # sma100
                dp_interval = math.ceil(100.0 / dp_per_sma)
                start = idx - 1 - input_length * dp_interval
                end = idx
                sma100 = df["SMA100"].iloc[start:end:dp_interval].values
                sma100 = utility.math_functions.price_to_percentage(sma100) * 2.0 / dp_interval
                sma100 = np.tanh(sma100)

                # sma200
                dp_interval = math.ceil(200.0 / dp_per_sma)
                start = idx - 1 - input_length * dp_interval
                end = idx
                sma200 = df["SMA200"].iloc[start:end:50].values
                sma200 = utility.math_functions.price_to_percentage(sma200) * 2.5 / dp_interval
                sma200 = np.tanh(sma200)

                # combine matrix
                xx = np.concatenate((close.reshape(-1, 1),
                                     sma5.reshape(-1, 1),
                                     sma20.reshape(-1, 1),
                                     sma100.reshape(-1, 1),
                                     sma200.reshape(-1, 1)), axis=1)

                # get target
                idx_sma20 = idx
                if idx_sma20 + 19 >= len(df):
                    idx_sma20 = len(df) - 20
                sma20_future = df["SMA20"].iloc[idx_sma20 + 19]
                price_now = df["Close"].iloc[idx - 1]
                date_now = df["Date"].iloc[idx - 1]
                y = (sma20_future / price_now - 1.0) * 100.0

                # mapping to around (-2, 2) region
                y = np.tanh(y / 10.0)

                # check shape and nan
                if xx.shape != (input_length, 5):
                    logging.error("Incorrect xx shape:" + str(xx.shape))
                elif np.count_nonzero(np.isnan(xx)) > 0:
                    logging.error("Found NAN in xx")
                elif y == np.nan:
                    logging.error("y = NAN")
                else:
                    self._x_train.append(xx.reshape((1, xx.shape[0], xx.shape[1])))
                    self._y_train.append(y)
                    self._date.append(date_now)
                    self._price_now.append(price_now)

        # need to convert the sample as first dimension
        self._x_train = np.concatenate(tuple(self._x_train))
        self._y_train = np.array(self._y_train).reshape(-1, 1)
        self._date = np.array(self._date).reshape(-1, 1)
        self._price_now = np.array(self._price_now).reshape(-1, 1)
        return self

    def split(self, test_size=0.2):
        """
        Split to training and testing
        :param test_size:
        :return:
        """
        if self._skip_to_end or self._single_ticker_call:
            return self

        self._x_train, self._x_test, self._y_train, self._y_test \
            = train_test_split(self._x_train, self._y_train, test_size=test_size, random_state=123)

        return self

    def get(self, save=False, separate_input=False):
        """
        :param save: save pickle data for faster access next time
        :param separate_input: used for multiple input pipes
        :return:
        """
        if save:
            with open("x_train.pk", "wb") as fp:
                pk.dump(self._x_train, fp)
            with open("x_test.pk", "wb") as fp:
                pk.dump(self._x_test, fp)
            with open("y_train.pk", "wb") as fp:
                pk.dump(self._y_train, fp)
            with open("y_test.pk", "wb") as fp:
                pk.dump(self._y_test, fp)

        if separate_input:
            self._x_train = [self._x_train[:, :, 0:1],
                             self._x_train[:, :, 1:2],
                             self._x_train[:, :, 2:3],
                             self._x_train[:, :, 3:4],
                             self._x_train[:, :, 4:5]]
            if not self._single_ticker_call:
                self._x_test = [self._x_test[:, :, 0:1],
                                self._x_test[:, :, 1:2],
                                self._x_test[:, :, 2:3],
                                self._x_test[:, :, 3:4],
                                self._x_test[:, :, 4:5]]

        if self._single_ticker_call:
            return self._x_train, self._x_test, self._y_train, self._y_test, self._date, self._price_now
        else:
            return self._x_train, self._x_test, self._y_train, self._y_test
