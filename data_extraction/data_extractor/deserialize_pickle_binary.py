import pickle
from data_extractor.log_and_discard_exceptions import log_and_discard_exceptions


@log_and_discard_exceptions
def decode_pickle_binary(fp):
    with open(fp, 'rb'):
        return pickle.load(fp, fix_imports=True, encoding="ASCII", errors="strict")
