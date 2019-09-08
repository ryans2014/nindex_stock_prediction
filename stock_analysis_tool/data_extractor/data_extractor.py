from abc import ABC, abstractmethod


class DataExtractor(ABC):

    @abstractmethod
    def extract(self, ticker: str, get_full_data: bool) -> dict:
        """
        returns a json-type dict object
        """
        pass

    def batch_extract(self, ticker_list: list, get_full_data: bool) -> list:
        """
        returns a list of json-type dict object
        """
        ret = []
        for ticker in ticker_list:
            data = self.extract(ticker, get_full_data)
            ret.append(data)
        return ret
