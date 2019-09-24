from time import time, sleep
from functools import wraps
import asyncio


def frequency_limiter(interval_by_second: float):

    def decorator_frequency_limiter(fun):

        @wraps(fun)
        def wrapper_function(*args, **kwargs):
            nonlocal _prev_time
            waiting_time = interval_by_second - (time() - _prev_time)
            if waiting_time > 0.0:
                sleep(waiting_time)
            _prev_time = time()
            return fun(*args, **kwargs)

        _prev_time = time() - interval_by_second
        return wrapper_function

    return decorator_frequency_limiter


def frequency_limiter_async(interval_by_second: float):

    def decorator_frequency_limiter(fun):

        @wraps(fun)
        async def wrapper_function(*args, **kwargs):
            nonlocal _prev_time
            waiting_time = interval_by_second - (time() - _prev_time)
            if waiting_time > 0.0:
                await asyncio.sleep(waiting_time)
            _prev_time = time()
            ret = await fun(*args, **kwargs)
            return ret

        _prev_time = time() - interval_by_second
        return wrapper_function

    return decorator_frequency_limiter
