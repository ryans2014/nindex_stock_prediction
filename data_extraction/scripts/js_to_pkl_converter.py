import os
import glob
import json
import pickle
from configuration.configuration import work_dir
from data_extractor.alphavantage_data_processor import process_data_to_get_closing_price
from utility.log_and_discard_exceptions import log_and_discard_exceptions


# read json file, get closing price, stored to pkl file (binary)
@log_and_discard_exceptions
def convert_raw_json_to_pkl():
    json_path = os.path.join(work_dir, "sp500_data_raw_json\\*.json")
    pkl_file_path = os.path.join(work_dir, "sp500_date_close_volume.pkl")
    data = {}
    for file in glob.glob(pathname=json_path, recursive=False):
        with open(file, 'r') as f:
            raw_obj = json.load(f)
            stk_obj = process_data_to_get_closing_price(raw_obj)
            data[stk_obj.ticker] = stk_obj
    with open(pkl_file_path, 'wb') as fp:
        pickle.dump(data, fp)


if __name__ == "__main__":
    convert_raw_json_to_pkl()
