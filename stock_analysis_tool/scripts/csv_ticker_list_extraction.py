import csv
from data_extractor import get_data


# get sp500 data, store json file in work dir, skip if json file already exists
if __name__ == "__main__":
    tickers = set()
    with open("pop_stk.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            tickers.add(line[0])

    # with open("index_list.csv", 'r') as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     for line in csv_reader:
    #         tickers.add(line[0])

    for ticker in tickers:
        get_data(ticker=ticker, force_update=False, save=True)
