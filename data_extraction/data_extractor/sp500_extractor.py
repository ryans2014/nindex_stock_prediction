from data_extractor.alphavantage import AlphavantageExtractor
import csv
import os
import json
import logging


# get sp500 data, store json file in work dir, skip if json file already exists
def get_sp500_full_history():
    extractor = AlphavantageExtractor()
    tickers = []
    path = os.getcwd()
    with open("sp500_list.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            tickers.append(line[0])
    for ticker in tickers:
        file_name = ticker.lower() + ".json"
        if os.path.isfile(os.path.join(path, file_name)):
            continue
        data = extractor.extract(ticker, True)
        if data == {}:
            continue
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file)
            logging.info("Got full history of %s" % ticker)


