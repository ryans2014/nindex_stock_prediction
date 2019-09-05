import functools


def single_instance_generator(fun):
    """
    :param fun: generator function with zero input
    :return:
        Initialize a generator function fun with zero argument
        Calling fun will return the next generated object
    """

    @functools.wraps(fun)
    def get_next_function():
        return next(generator_object)

    generator_object = fun()
    return get_next_function
