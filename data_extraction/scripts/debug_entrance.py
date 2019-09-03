from data_extractor import Extractor
import sys
import logging


# test entrance
def debug_entrance():
    extractor = Extractor()
    return extractor.batch_extract(['AAPL'], True)


if __name__ == "__main__":
    debug_entrance()
