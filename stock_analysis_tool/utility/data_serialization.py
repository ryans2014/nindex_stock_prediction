import pickle
from utility.generic_exception_handler import log_and_discard_exceptions


@log_and_discard_exceptions
def decode_pickle_binary(file_path):
    with open(file_path, 'rb') as fp:
        return pickle.load(fp, fix_imports=True, encoding="ASCII", errors="strict")


@log_and_discard_exceptions
def encode_pickle_binary(obj, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)
