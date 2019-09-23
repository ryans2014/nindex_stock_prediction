import configuration
import os
import json
from .data_extractor import DataExtractor


class CacheExtractor(DataExtractor):

    def __init__(self, cache_folder_name="raw_json_cache"):
        self._cache_path = os.path.join(os.getcwd(), cache_folder_name)

    def extract(self, ticker: str, get_full_data: bool = True) -> dict:
        if not ticker.lower().endswith("json"):
            ticker = ticker + ".json"
        file_path = os.path.join(self._cache_path, ticker.lower())
        if not os.path.isfile(file_path):
            raise ValueError("Incorrect file path")
        with open(file_path, "r") as fp:
            return json.load(fp)


class CacheWriter:

    def __init__(self, cache_folder_name="raw_json_cache"):
        self._cache_path = os.path.join(os.getcwd(), cache_folder_name)

    def write(self, ticker: str, obj: dict):
        if not ticker.lower().endswith("json"):
            ticker = ticker + ".json"
        file_path = os.path.join(self._cache_path, ticker.lower())
        with open(file_path, "w") as fp:
            return json.dump(obj, fp)
