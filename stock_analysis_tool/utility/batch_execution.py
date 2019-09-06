from functools import wraps


def batch_execution(fun):
    """
    :param fun: function to be decorated
        If any input parameter to fun is a list, tuple, dict, or set, it will be unpacked and executed indiviaually
    """

    @wraps(fun)
    def decorator_function(*args, **kwargs):
        if len(args) != 2 or len(kwargs) != 0:
            raise ValueError("Only support zero kwargs! Only support class member function with 1 argument!")
        if type(args[1]) is dict:
            ret = []
            for k, v in args[1].items:
                ret.append(fun(args[0], v))
            return ret
        if type(args[1]) in [list, set, tuple]:
            ret = []
            for unit_arg in args[1]:
                ret.append(fun(args[0], unit_arg))
            return ret
        else:
            return fun(*args, **kwargs)

    return decorator_function
