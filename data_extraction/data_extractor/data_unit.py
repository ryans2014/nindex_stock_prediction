from data_extractor import date as dt
import logging
from enum import Enum
import pickle


class DataType(Enum):
    Date_Close_Volume = 1


class StockData:

    def __init__(self, ticker: str, data_type: DataType):
        self.ticker = ticker
        self.data = []
        self.type = data_type
        self.start_date = 999999
        self.end_date = 0

    def append(self, entry: tuple):
        """
        :param entry: tuple(int date + float price)
        :return:
        """
        if type(entry) is not tuple or len(entry) < 2:
            logging.warning("StockDate insert need 2+ element tuples")
            return
        if dt.validate(entry[0]) is False:
            logging.warning("Date validation failed: %d" % entry[0])
            return
        self.data.append(entry)
        if entry[0] < self.start_date:
            self.start_date = entry[0]
        if entry[0] > self.end_date:
            self.end_date = entry[0]

    def sort(self):
        self.data.sort()

    def dump(self, file_name=""):
        if file_name == "":
            file_name = self._ticker + ".pkl"
        with open(file_name, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, file_name: str):
        with open(file_name, 'rb') as fp:
            return pickle.load(fp)
