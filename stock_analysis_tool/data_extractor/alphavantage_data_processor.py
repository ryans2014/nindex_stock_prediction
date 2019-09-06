import utility
import logging
import pandas as pd
import json


@utility.log_and_discard_exceptions
def convert_alphavantage_data_to_pandas(js_obj):
    """
    :param js_obj: json file path string or json dict object
    :return: (ticker string, pandas DataFrame)
    """
    if type(js_obj) is str:
        with open(js_obj) as fp:
            js_obj = json.load(fp)

    ticker = js_obj['Meta Data']['2. Symbol']
    js_obj = js_obj['Time Series (Daily)']
    logging.info("process_data_to_get_closing_price: %s" % ticker)
    logging.getLogger().handlers[0].flush()
    ls_obj = []
    for date, price in js_obj.items():
        int_date = utility.date.date_to_int(date)
        adjusted_close = float(price['5. adjusted close'])
        volume = float(price['6. volume'])
        ls_obj.append([int_date, adjusted_close, volume])
    df = pd.DataFrame(data=ls_obj, columns=["Date", "Close", "Volume"])
    df.sort_values(by="Date", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return ticker, df
