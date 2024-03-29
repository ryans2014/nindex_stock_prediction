import functools


def named_model(fun):
    """
    :param fun: function to generate keras models. Need to return a keras model.
    :return: model.cname = fun.__name__
    Side effect: insert model name to map
    """
    @functools.wraps(fun)
    def call_and_set_cname():
        model = fun()
        model_name = fun.__name__
        model.cname = model_name
        return model

    return call_and_set_cname


def multi_input_model(fun):
    """
    :param fun: function to generate keras models. Need to return a keras model.
    :return: model.multi_input = True
    """
    @functools.wraps(fun)
    def call_and_set_flag():
        model = fun()
        model.multi_input = True
        return model

    return call_and_set_flag
