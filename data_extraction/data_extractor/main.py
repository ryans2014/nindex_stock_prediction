import data_extractor.configuration
from data_extractor.alphavantage import AlphavantageExtractor
from sp500_extractor import get_sp500_full_history
import sys
import logging


# test entrance
def debug_entrance():
    extractor = AlphavantageExtractor()
    return extractor.batch_extract(['AAPLBB'], True)


if __name__ == "__main__":
    for i, arg in enumerate(sys.argv):
        logging.info("Input argument(%d): %s" % (i, arg))
    if len(sys.argv) <= 1:
        logging.error("Calling from main.py. Insufficient command argument.")
    elif sys.argv[1].lower() == "debug":
        logging.info("Calling from main.py. Arg = test")
        debug_entrance()
    elif sys.argv[1].lower() == "sp500":
        logging.info("Calling from main.py. Arg = sp500")
        get_sp500_full_history()
    else:
        logging.error("Unsupported command argument %s" % sys.argv[1])
