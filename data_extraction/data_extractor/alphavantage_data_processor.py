from data_extractor.data_unit import StockData, DataType
from data_extractor.date import date_to_int
from data_extractor.log_and_discard_exceptions import log_and_discard_exceptions
import logging


@log_and_discard_exceptions
def process_data_to_get_closing_price(js_obj: dict) -> StockData:

    ticker = js_obj['Meta Data']['2. Symbol']
    stk_obj = StockData(ticker=ticker, data_type=DataType.Date_Close_Volume)
    js_obj = js_obj['Time Series (Daily)']

    logging.info("process_data_to_get_closing_price: %s" % ticker)
    logging.getLogger().handlers[0].flush()

    for date, price in js_obj.items():

        int_date = date_to_int(date)
        adjusted_close = float(price['5. adjusted close'])
        volume = float(price['6. volume'])
        stk_obj.append((int_date, adjusted_close, volume))

        # split_coef = float(price['8. split coefficient'])
        # if abs(split_coef - 1.0) > 0.1:
        #     logging.warning("Found split on ticker %s, date %s" % (ticker, date))

    stk_obj.sort()
    return stk_obj

