from data_extractor.alphavantage_extractor import AlphavantageExtractor
import sys
import logging


# test entrance
def debug_entrance():
    extractor = AlphavantageExtractor()
    return extractor.batch_extract(['AAPL'], True)


if __name__ == "__main__":
    debug_entrance()
