from datetime import datetime, timedelta
import logging

_ref_date = datetime.strptime('1970-1-1', "%Y-%m-%d")
_ref_weekday = _ref_date.weekday()


def date_to_int(date_str: str) -> int:
    """ convert date string to int """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exp:
        logging.warning("Date string cannot be parsed: %s" % date_str)
        raise exp
    delta = date_obj - _ref_date
    return delta.days


def int_to_date(date_int: int):
    """ convert int to date object """
    delta = timedelta(days=date_int)
    return _ref_date + delta


def weekday(date_obj) -> int:
    """
    :param date_obj: string or int
    :return: weekdays from 1 to 7
    """
    if type(date_obj) is str:
        date_obj = date_to_int(date_obj)
    return (date_obj + _ref_weekday) % 7


def validate(date_obj) -> bool:
    if type(date_obj) is str:
        date_obj = date_to_int(date_obj)
    if date_obj <= 0:
        return False
    today = datetime.now() - _ref_date
    today = today.days + 1
    if date_obj > today:
        return False
    return True
