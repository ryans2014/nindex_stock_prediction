from functools import wraps
import logging


def log_and_discard_exceptions(fun):
    """
        Wraps fun. If exceptions is caught, log the error and return None
    """

    @wraps(fun)
    def decorator_function(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except BaseException as e:
            fun_name = fun.__module__ + "." + fun.__name__
            logging.warning("In function %s, %s is caught and discarded: %s" % (fun_name, type(e).__name__, str(e)))

    return decorator_function
